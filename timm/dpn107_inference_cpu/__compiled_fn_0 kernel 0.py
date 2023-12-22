
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
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
                                auto tmp12 = masked_load(in_out_ptr0 + static_cast<long>((-14464L) + x3 + (256L*x2) + (28672L*x1) + (1605632L*x0)), to_float_mask(tmp10));
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
                                auto tmp20 = masked_load(in_out_ptr0 + static_cast<long>((-14336L) + x3 + (256L*x2) + (28672L*x1) + (1605632L*x0)), to_float_mask(tmp18));
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
                                auto tmp29 = masked_load(in_out_ptr0 + static_cast<long>((-14208L) + x3 + (256L*x2) + (28672L*x1) + (1605632L*x0)), to_float_mask(tmp27));
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
                                auto tmp38 = masked_load(in_out_ptr0 + static_cast<long>((-128L) + x3 + (256L*x2) + (28672L*x1) + (1605632L*x0)), to_float_mask(tmp36));
                                return tmp38;
                            }
                            ;
                            auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp37(), to_float_mask(tmp36));
                            auto tmp40 = at::vec::maximum(tmp39, tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(in_out_ptr0 + static_cast<long>(x3 + (256L*x2) + (28672L*x1) + (1605632L*x0)), to_float_mask(tmp41));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp42(), to_float_mask(tmp41));
                            auto tmp45 = at::vec::maximum(tmp44, tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = masked_load(in_out_ptr0 + static_cast<long>(128L + x3 + (256L*x2) + (28672L*x1) + (1605632L*x0)), to_float_mask(tmp46));
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
                                auto tmp57 = masked_load(in_out_ptr0 + static_cast<long>(14208L + x3 + (256L*x2) + (28672L*x1) + (1605632L*x0)), to_float_mask(tmp55));
                                return tmp57;
                            }
                            ;
                            auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp56(), to_float_mask(tmp55));
                            auto tmp59 = at::vec::maximum(tmp58, tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = masked_load(in_out_ptr0 + static_cast<long>(14336L + x3 + (256L*x2) + (28672L*x1) + (1605632L*x0)), to_float_mask(tmp60));
                                return tmp62;
                            }
                            ;
                            auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp61(), to_float_mask(tmp60));
                            auto tmp64 = at::vec::maximum(tmp63, tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = masked_load(in_out_ptr0 + static_cast<long>(14464L + x3 + (256L*x2) + (28672L*x1) + (1605632L*x0)), to_float_mask(tmp65));
                                return tmp67;
                            }
                            ;
                            auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp66(), to_float_mask(tmp65));
                            auto tmp69 = at::vec::maximum(tmp68, tmp64);
                            tmp69.store(out_ptr0 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (401408L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                        auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(0.001);
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
                        auto tmp19 = tmp0 - tmp18;
                        auto tmp21 = tmp20 + tmp5;
                        auto tmp22 = tmp21.sqrt();
                        auto tmp23 = tmp22.reciprocal();
                        auto tmp24 = tmp23 * tmp10;
                        auto tmp25 = tmp19 * tmp24;
                        auto tmp27 = tmp25 * tmp26;
                        auto tmp29 = tmp27 + tmp28;
                        auto tmp30 = at::vec::clamp_min(tmp29, decltype(tmp29)(0));
                        tmp17.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (401408L*x0)));
                        tmp30.store(out_ptr2 + static_cast<long>(x1 + (128L*x2) + (401408L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(200L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(316L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr2[static_cast<long>(x1)];
                    auto tmp33 = in_ptr3[static_cast<long>(x1)];
                    auto tmp41 = in_ptr4[static_cast<long>(x1)];
                    auto tmp43 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(256);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (296L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (276L*x0))];
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        return tmp8;
                    }
                    ;
                    auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp10 = tmp0 >= tmp3;
                    auto tmp11 = static_cast<long>(316);
                    auto tmp12 = tmp0 < tmp11;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = c10::convert<long>((-256L) + x1);
                        auto tmp15 = static_cast<long>(0);
                        auto tmp16 = tmp14 >= tmp15;
                        auto tmp17 = static_cast<long>(40);
                        auto tmp18 = tmp14 < tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = in_ptr0[static_cast<long>(x1 + (296L*x0))];
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp22 = tmp14 >= tmp17;
                        auto tmp23 = static_cast<long>(60);
                        auto tmp24 = tmp14 < tmp23;
                        auto tmp25 = [&]
                        {
                            auto tmp26 = in_ptr1[static_cast<long>((-40L) + x1 + (276L*x0))];
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                        auto tmp28 = tmp18 ? tmp21 : tmp27;
                        return tmp28;
                    }
                    ;
                    auto tmp29 = tmp10 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp30 = tmp4 ? tmp9 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(0.001);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    auto tmp45 = tmp44 * (tmp44>0);
                    out_ptr0[static_cast<long>(x1 + (316L*x0))] = tmp45;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_5 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(200L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    auto tmp47 = in_ptr3[static_cast<long>(x1)];
                    auto tmp49 = in_ptr4[static_cast<long>(x1)];
                    auto tmp57 = in_ptr5[static_cast<long>(x1)];
                    auto tmp59 = in_ptr6[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(256);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (296L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (276L*x0))];
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = in_ptr2[static_cast<long>(x1 + (276L*x0))];
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(336);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = c10::convert<long>((-256L) + x1);
                        auto tmp17 = static_cast<long>(0);
                        auto tmp18 = tmp16 >= tmp17;
                        auto tmp19 = static_cast<long>(60);
                        auto tmp20 = tmp16 < tmp19;
                        auto tmp21 = [&]
                        {
                            auto tmp22 = c10::convert<long>((-256L) + x1);
                            auto tmp23 = static_cast<long>(0);
                            auto tmp24 = tmp22 >= tmp23;
                            auto tmp25 = static_cast<long>(40);
                            auto tmp26 = tmp22 < tmp25;
                            auto tmp27 = [&]
                            {
                                auto tmp28 = in_ptr0[static_cast<long>(x1 + (296L*x0))];
                                return tmp28;
                            }
                            ;
                            auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                            auto tmp30 = tmp22 >= tmp25;
                            auto tmp31 = static_cast<long>(60);
                            auto tmp32 = tmp22 < tmp31;
                            auto tmp33 = [&]
                            {
                                auto tmp34 = in_ptr1[static_cast<long>((-40L) + x1 + (276L*x0))];
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp30 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                            auto tmp36 = tmp26 ? tmp29 : tmp35;
                            return tmp36;
                        }
                        ;
                        auto tmp37 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                        auto tmp38 = tmp16 >= tmp19;
                        auto tmp39 = static_cast<long>(80);
                        auto tmp40 = tmp16 < tmp39;
                        auto tmp41 = [&]
                        {
                            auto tmp42 = in_ptr2[static_cast<long>((-60L) + x1 + (276L*x0))];
                            return tmp42;
                        }
                        ;
                        auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                        auto tmp44 = tmp20 ? tmp37 : tmp43;
                        return tmp44;
                    }
                    ;
                    auto tmp45 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp46 = tmp4 ? tmp11 : tmp45;
                    auto tmp48 = decltype(tmp46)(tmp46 - tmp47);
                    auto tmp50 = static_cast<float>(0.001);
                    auto tmp51 = decltype(tmp49)(tmp49 + tmp50);
                    auto tmp52 = std::sqrt(tmp51);
                    auto tmp53 = 1 / tmp52;
                    auto tmp54 = static_cast<float>(1.0);
                    auto tmp55 = decltype(tmp53)(tmp53 * tmp54);
                    auto tmp56 = decltype(tmp48)(tmp48 * tmp55);
                    auto tmp58 = decltype(tmp56)(tmp56 * tmp57);
                    auto tmp60 = decltype(tmp58)(tmp58 + tmp59);
                    auto tmp61 = tmp60 * (tmp60>0);
                    in_out_ptr0[static_cast<long>(x1 + (336L*x0))] = tmp61;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_8 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(200L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_10 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(356L); x1+=static_cast<long>(1L))
                {
                    auto tmp63 = in_ptr4[static_cast<long>(x1)];
                    auto tmp65 = in_ptr5[static_cast<long>(x1)];
                    auto tmp73 = in_ptr6[static_cast<long>(x1)];
                    auto tmp75 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(256);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (296L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (276L*x0))];
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = in_ptr2[static_cast<long>(x1 + (276L*x0))];
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = in_ptr3[static_cast<long>(x1 + (276L*x0))];
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp14 = tmp0 >= tmp3;
                    auto tmp15 = static_cast<long>(356);
                    auto tmp16 = tmp0 < tmp15;
                    auto tmp17 = [&]
                    {
                        auto tmp18 = c10::convert<long>((-256L) + x1);
                        auto tmp19 = static_cast<long>(0);
                        auto tmp20 = tmp18 >= tmp19;
                        auto tmp21 = static_cast<long>(80);
                        auto tmp22 = tmp18 < tmp21;
                        auto tmp23 = [&]
                        {
                            auto tmp24 = c10::convert<long>((-256L) + x1);
                            auto tmp25 = static_cast<long>(0);
                            auto tmp26 = tmp24 >= tmp25;
                            auto tmp27 = static_cast<long>(60);
                            auto tmp28 = tmp24 < tmp27;
                            auto tmp29 = [&]
                            {
                                auto tmp30 = c10::convert<long>((-256L) + x1);
                                auto tmp31 = static_cast<long>(0);
                                auto tmp32 = tmp30 >= tmp31;
                                auto tmp33 = static_cast<long>(40);
                                auto tmp34 = tmp30 < tmp33;
                                auto tmp35 = [&]
                                {
                                    auto tmp36 = in_ptr0[static_cast<long>(x1 + (296L*x0))];
                                    return tmp36;
                                }
                                ;
                                auto tmp37 = tmp34 ? tmp35() : static_cast<decltype(tmp35())>(0.0);
                                auto tmp38 = tmp30 >= tmp33;
                                auto tmp39 = static_cast<long>(60);
                                auto tmp40 = tmp30 < tmp39;
                                auto tmp41 = [&]
                                {
                                    auto tmp42 = in_ptr1[static_cast<long>((-40L) + x1 + (276L*x0))];
                                    return tmp42;
                                }
                                ;
                                auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                                auto tmp44 = tmp34 ? tmp37 : tmp43;
                                return tmp44;
                            }
                            ;
                            auto tmp45 = tmp28 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                            auto tmp46 = tmp24 >= tmp27;
                            auto tmp47 = static_cast<long>(80);
                            auto tmp48 = tmp24 < tmp47;
                            auto tmp49 = [&]
                            {
                                auto tmp50 = in_ptr2[static_cast<long>((-60L) + x1 + (276L*x0))];
                                return tmp50;
                            }
                            ;
                            auto tmp51 = tmp46 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                            auto tmp52 = tmp28 ? tmp45 : tmp51;
                            return tmp52;
                        }
                        ;
                        auto tmp53 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                        auto tmp54 = tmp18 >= tmp21;
                        auto tmp55 = static_cast<long>(100);
                        auto tmp56 = tmp18 < tmp55;
                        auto tmp57 = [&]
                        {
                            auto tmp58 = in_ptr3[static_cast<long>((-80L) + x1 + (276L*x0))];
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp54 ? tmp57() : static_cast<decltype(tmp57())>(0.0);
                        auto tmp60 = tmp22 ? tmp53 : tmp59;
                        return tmp60;
                    }
                    ;
                    auto tmp61 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                    auto tmp62 = tmp4 ? tmp13 : tmp61;
                    auto tmp64 = decltype(tmp62)(tmp62 - tmp63);
                    auto tmp66 = static_cast<float>(0.001);
                    auto tmp67 = decltype(tmp65)(tmp65 + tmp66);
                    auto tmp68 = std::sqrt(tmp67);
                    auto tmp69 = 1 / tmp68;
                    auto tmp70 = static_cast<float>(1.0);
                    auto tmp71 = decltype(tmp69)(tmp69 * tmp70);
                    auto tmp72 = decltype(tmp64)(tmp64 * tmp71);
                    auto tmp74 = decltype(tmp72)(tmp72 * tmp73);
                    auto tmp76 = decltype(tmp74)(tmp74 + tmp75);
                    auto tmp77 = tmp76 * (tmp76>0);
                    in_out_ptr0[static_cast<long>(x1 + (356L*x0))] = tmp77;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_11 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(200L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_13 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(100);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = c10::convert<long>(x1);
                        auto tmp7 = static_cast<long>(0);
                        auto tmp8 = tmp6 >= tmp7;
                        auto tmp9 = static_cast<long>(80);
                        auto tmp10 = tmp6 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = c10::convert<long>(x1);
                            auto tmp13 = static_cast<long>(0);
                            auto tmp14 = tmp12 >= tmp13;
                            auto tmp15 = static_cast<long>(60);
                            auto tmp16 = tmp12 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = c10::convert<long>(x1);
                                auto tmp19 = static_cast<long>(0);
                                auto tmp20 = tmp18 >= tmp19;
                                auto tmp21 = static_cast<long>(40);
                                auto tmp22 = tmp18 < tmp21;
                                auto tmp23 = [&]
                                {
                                    auto tmp24 = in_ptr0[static_cast<long>(256L + x1 + (296L*x0))];
                                    return tmp24;
                                }
                                ;
                                auto tmp25 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                                auto tmp26 = tmp18 >= tmp21;
                                auto tmp27 = static_cast<long>(60);
                                auto tmp28 = tmp18 < tmp27;
                                auto tmp29 = [&]
                                {
                                    auto tmp30 = in_ptr1[static_cast<long>(216L + x1 + (276L*x0))];
                                    return tmp30;
                                }
                                ;
                                auto tmp31 = tmp26 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                                auto tmp32 = tmp22 ? tmp25 : tmp31;
                                return tmp32;
                            }
                            ;
                            auto tmp33 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp34 = tmp12 >= tmp15;
                            auto tmp35 = static_cast<long>(80);
                            auto tmp36 = tmp12 < tmp35;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = in_ptr2[static_cast<long>(196L + x1 + (276L*x0))];
                                return tmp38;
                            }
                            ;
                            auto tmp39 = tmp34 ? tmp37() : static_cast<decltype(tmp37())>(0.0);
                            auto tmp40 = tmp16 ? tmp33 : tmp39;
                            return tmp40;
                        }
                        ;
                        auto tmp41 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp42 = tmp6 >= tmp9;
                        auto tmp43 = static_cast<long>(100);
                        auto tmp44 = tmp6 < tmp43;
                        auto tmp45 = [&]
                        {
                            auto tmp46 = in_ptr3[static_cast<long>(176L + x1 + (276L*x0))];
                            return tmp46;
                        }
                        ;
                        auto tmp47 = tmp42 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                        auto tmp48 = tmp10 ? tmp41 : tmp47;
                        return tmp48;
                    }
                    ;
                    auto tmp49 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp50 = tmp0 >= tmp3;
                    auto tmp51 = static_cast<long>(120);
                    auto tmp52 = tmp0 < tmp51;
                    auto tmp53 = [&]
                    {
                        auto tmp54 = in_ptr4[static_cast<long>(156L + x1 + (276L*x0))];
                        return tmp54;
                    }
                    ;
                    auto tmp55 = tmp50 ? tmp53() : static_cast<decltype(tmp53())>(0.0);
                    auto tmp56 = tmp4 ? tmp49 : tmp55;
                    out_ptr0[static_cast<long>(x1 + (120L*x0))] = tmp56;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(376L); x1+=static_cast<long>(1L))
                {
                    auto tmp23 = in_ptr5[static_cast<long>(x1)];
                    auto tmp25 = in_ptr6[static_cast<long>(x1)];
                    auto tmp33 = in_ptr7[static_cast<long>(x1)];
                    auto tmp35 = in_ptr8[static_cast<long>(x1)];
                    auto tmp38 = in_ptr9[static_cast<long>(x1)];
                    auto tmp40 = in_ptr10[static_cast<long>(x1)];
                    auto tmp46 = in_ptr11[static_cast<long>(x1)];
                    auto tmp48 = in_ptr12[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(256);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (296L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (276L*x0))];
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = in_ptr2[static_cast<long>(x1 + (276L*x0))];
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = in_ptr3[static_cast<long>(x1 + (276L*x0))];
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = in_ptr4[static_cast<long>(x1 + (276L*x0))];
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp16 = tmp0 >= tmp3;
                    auto tmp17 = static_cast<long>(376);
                    auto tmp18 = tmp0 < tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = out_ptr0[static_cast<long>((-256L) + x1 + (120L*x0))];
                        return tmp20;
                    }
                    ;
                    auto tmp21 = tmp16 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp22 = tmp4 ? tmp15 : tmp21;
                    auto tmp24 = decltype(tmp22)(tmp22 - tmp23);
                    auto tmp26 = static_cast<float>(0.001);
                    auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                    auto tmp28 = std::sqrt(tmp27);
                    auto tmp29 = 1 / tmp28;
                    auto tmp30 = static_cast<float>(1.0);
                    auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                    auto tmp32 = decltype(tmp24)(tmp24 * tmp31);
                    auto tmp34 = decltype(tmp32)(tmp32 * tmp33);
                    auto tmp36 = decltype(tmp34)(tmp34 + tmp35);
                    auto tmp37 = tmp36 * (tmp36>0);
                    auto tmp39 = decltype(tmp22)(tmp22 - tmp38);
                    auto tmp41 = decltype(tmp40)(tmp40 + tmp26);
                    auto tmp42 = std::sqrt(tmp41);
                    auto tmp43 = 1 / tmp42;
                    auto tmp44 = decltype(tmp43)(tmp43 * tmp30);
                    auto tmp45 = decltype(tmp39)(tmp39 * tmp44);
                    auto tmp47 = decltype(tmp45)(tmp45 * tmp46);
                    auto tmp49 = decltype(tmp47)(tmp47 + tmp48);
                    auto tmp50 = tmp49 * (tmp49>0);
                    out_ptr2[static_cast<long>(x1 + (376L*x0))] = tmp37;
                    out_ptr3[static_cast<long>(x1 + (376L*x0))] = tmp50;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_14 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(704L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr2[static_cast<long>(x1)];
                    auto tmp33 = in_ptr3[static_cast<long>(x1)];
                    auto tmp41 = in_ptr4[static_cast<long>(x1)];
                    auto tmp43 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(512);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (640L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (576L*x0))];
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        return tmp8;
                    }
                    ;
                    auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp10 = tmp0 >= tmp3;
                    auto tmp11 = static_cast<long>(704);
                    auto tmp12 = tmp0 < tmp11;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = c10::convert<long>((-512L) + x1);
                        auto tmp15 = static_cast<long>(0);
                        auto tmp16 = tmp14 >= tmp15;
                        auto tmp17 = static_cast<long>(128);
                        auto tmp18 = tmp14 < tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = in_ptr0[static_cast<long>(x1 + (640L*x0))];
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp22 = tmp14 >= tmp17;
                        auto tmp23 = static_cast<long>(192);
                        auto tmp24 = tmp14 < tmp23;
                        auto tmp25 = [&]
                        {
                            auto tmp26 = in_ptr1[static_cast<long>((-128L) + x1 + (576L*x0))];
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                        auto tmp28 = tmp18 ? tmp21 : tmp27;
                        return tmp28;
                    }
                    ;
                    auto tmp29 = tmp10 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp30 = tmp4 ? tmp9 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(0.001);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    auto tmp45 = tmp44 * (tmp44>0);
                    out_ptr0[static_cast<long>(x1 + (704L*x0))] = tmp45;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp47 = in_ptr3[static_cast<long>(x1)];
                    auto tmp49 = in_ptr4[static_cast<long>(x1)];
                    auto tmp57 = in_ptr5[static_cast<long>(x1)];
                    auto tmp59 = in_ptr6[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(512);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (640L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (576L*x0))];
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = in_ptr2[static_cast<long>(x1 + (576L*x0))];
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(768);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = c10::convert<long>((-512L) + x1);
                        auto tmp17 = static_cast<long>(0);
                        auto tmp18 = tmp16 >= tmp17;
                        auto tmp19 = static_cast<long>(192);
                        auto tmp20 = tmp16 < tmp19;
                        auto tmp21 = [&]
                        {
                            auto tmp22 = c10::convert<long>((-512L) + x1);
                            auto tmp23 = static_cast<long>(0);
                            auto tmp24 = tmp22 >= tmp23;
                            auto tmp25 = static_cast<long>(128);
                            auto tmp26 = tmp22 < tmp25;
                            auto tmp27 = [&]
                            {
                                auto tmp28 = in_ptr0[static_cast<long>(x1 + (640L*x0))];
                                return tmp28;
                            }
                            ;
                            auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                            auto tmp30 = tmp22 >= tmp25;
                            auto tmp31 = static_cast<long>(192);
                            auto tmp32 = tmp22 < tmp31;
                            auto tmp33 = [&]
                            {
                                auto tmp34 = in_ptr1[static_cast<long>((-128L) + x1 + (576L*x0))];
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp30 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                            auto tmp36 = tmp26 ? tmp29 : tmp35;
                            return tmp36;
                        }
                        ;
                        auto tmp37 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                        auto tmp38 = tmp16 >= tmp19;
                        auto tmp39 = static_cast<long>(256);
                        auto tmp40 = tmp16 < tmp39;
                        auto tmp41 = [&]
                        {
                            auto tmp42 = in_ptr2[static_cast<long>((-192L) + x1 + (576L*x0))];
                            return tmp42;
                        }
                        ;
                        auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                        auto tmp44 = tmp20 ? tmp37 : tmp43;
                        return tmp44;
                    }
                    ;
                    auto tmp45 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp46 = tmp4 ? tmp11 : tmp45;
                    auto tmp48 = decltype(tmp46)(tmp46 - tmp47);
                    auto tmp50 = static_cast<float>(0.001);
                    auto tmp51 = decltype(tmp49)(tmp49 + tmp50);
                    auto tmp52 = std::sqrt(tmp51);
                    auto tmp53 = 1 / tmp52;
                    auto tmp54 = static_cast<float>(1.0);
                    auto tmp55 = decltype(tmp53)(tmp53 * tmp54);
                    auto tmp56 = decltype(tmp48)(tmp48 * tmp55);
                    auto tmp58 = decltype(tmp56)(tmp56 * tmp57);
                    auto tmp60 = decltype(tmp58)(tmp58 + tmp59);
                    auto tmp61 = tmp60 * (tmp60>0);
                    in_out_ptr0[static_cast<long>(x1 + (768L*x0))] = tmp61;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_20 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_22 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(832L); x1+=static_cast<long>(1L))
                {
                    auto tmp63 = in_ptr4[static_cast<long>(x1)];
                    auto tmp65 = in_ptr5[static_cast<long>(x1)];
                    auto tmp73 = in_ptr6[static_cast<long>(x1)];
                    auto tmp75 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(512);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (640L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (576L*x0))];
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = in_ptr2[static_cast<long>(x1 + (576L*x0))];
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = in_ptr3[static_cast<long>(x1 + (576L*x0))];
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp14 = tmp0 >= tmp3;
                    auto tmp15 = static_cast<long>(832);
                    auto tmp16 = tmp0 < tmp15;
                    auto tmp17 = [&]
                    {
                        auto tmp18 = c10::convert<long>((-512L) + x1);
                        auto tmp19 = static_cast<long>(0);
                        auto tmp20 = tmp18 >= tmp19;
                        auto tmp21 = static_cast<long>(256);
                        auto tmp22 = tmp18 < tmp21;
                        auto tmp23 = [&]
                        {
                            auto tmp24 = c10::convert<long>((-512L) + x1);
                            auto tmp25 = static_cast<long>(0);
                            auto tmp26 = tmp24 >= tmp25;
                            auto tmp27 = static_cast<long>(192);
                            auto tmp28 = tmp24 < tmp27;
                            auto tmp29 = [&]
                            {
                                auto tmp30 = c10::convert<long>((-512L) + x1);
                                auto tmp31 = static_cast<long>(0);
                                auto tmp32 = tmp30 >= tmp31;
                                auto tmp33 = static_cast<long>(128);
                                auto tmp34 = tmp30 < tmp33;
                                auto tmp35 = [&]
                                {
                                    auto tmp36 = in_ptr0[static_cast<long>(x1 + (640L*x0))];
                                    return tmp36;
                                }
                                ;
                                auto tmp37 = tmp34 ? tmp35() : static_cast<decltype(tmp35())>(0.0);
                                auto tmp38 = tmp30 >= tmp33;
                                auto tmp39 = static_cast<long>(192);
                                auto tmp40 = tmp30 < tmp39;
                                auto tmp41 = [&]
                                {
                                    auto tmp42 = in_ptr1[static_cast<long>((-128L) + x1 + (576L*x0))];
                                    return tmp42;
                                }
                                ;
                                auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                                auto tmp44 = tmp34 ? tmp37 : tmp43;
                                return tmp44;
                            }
                            ;
                            auto tmp45 = tmp28 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                            auto tmp46 = tmp24 >= tmp27;
                            auto tmp47 = static_cast<long>(256);
                            auto tmp48 = tmp24 < tmp47;
                            auto tmp49 = [&]
                            {
                                auto tmp50 = in_ptr2[static_cast<long>((-192L) + x1 + (576L*x0))];
                                return tmp50;
                            }
                            ;
                            auto tmp51 = tmp46 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                            auto tmp52 = tmp28 ? tmp45 : tmp51;
                            return tmp52;
                        }
                        ;
                        auto tmp53 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                        auto tmp54 = tmp18 >= tmp21;
                        auto tmp55 = static_cast<long>(320);
                        auto tmp56 = tmp18 < tmp55;
                        auto tmp57 = [&]
                        {
                            auto tmp58 = in_ptr3[static_cast<long>((-256L) + x1 + (576L*x0))];
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp54 ? tmp57() : static_cast<decltype(tmp57())>(0.0);
                        auto tmp60 = tmp22 ? tmp53 : tmp59;
                        return tmp60;
                    }
                    ;
                    auto tmp61 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                    auto tmp62 = tmp4 ? tmp13 : tmp61;
                    auto tmp64 = decltype(tmp62)(tmp62 - tmp63);
                    auto tmp66 = static_cast<float>(0.001);
                    auto tmp67 = decltype(tmp65)(tmp65 + tmp66);
                    auto tmp68 = std::sqrt(tmp67);
                    auto tmp69 = 1 / tmp68;
                    auto tmp70 = static_cast<float>(1.0);
                    auto tmp71 = decltype(tmp69)(tmp69 * tmp70);
                    auto tmp72 = decltype(tmp64)(tmp64 * tmp71);
                    auto tmp74 = decltype(tmp72)(tmp72 * tmp73);
                    auto tmp76 = decltype(tmp74)(tmp74 + tmp75);
                    auto tmp77 = tmp76 * (tmp76>0);
                    in_out_ptr0[static_cast<long>(x1 + (832L*x0))] = tmp77;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_23 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_relu_25 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp9[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (640L*x1) + (640L*x1_inner) + (501760L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (576L*x1) + (576L*x1_inner) + (451584L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (576L*x1) + (576L*x1_inner) + (451584L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (576L*x1) + (576L*x1_inner) + (451584L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (576L*x1) + (576L*x1_inner) + (451584L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr0 + static_cast<long>(x1 + (784L*x2) + (702464L*x0)), static_cast<long>(784L));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(320);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = c10::convert<long>(x2);
                            auto tmp7 = static_cast<long>(0);
                            auto tmp8 = tmp6 >= tmp7;
                            auto tmp9 = static_cast<long>(256);
                            auto tmp10 = tmp6 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(x2);
                                auto tmp13 = static_cast<long>(0);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = static_cast<long>(192);
                                auto tmp16 = tmp12 < tmp15;
                                auto tmp17 = [&]
                                {
                                    auto tmp18 = c10::convert<long>(x2);
                                    auto tmp19 = static_cast<long>(0);
                                    auto tmp20 = tmp18 >= tmp19;
                                    auto tmp21 = static_cast<long>(128);
                                    auto tmp22 = tmp18 < tmp21;
                                    auto tmp23 = [&]
                                    {
                                        auto tmp24 = in_ptr0[static_cast<long>(512L + x2 + (640L*x1) + (501760L*x0))];
                                        return tmp24;
                                    }
                                    ;
                                    auto tmp25 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                                    auto tmp26 = tmp18 >= tmp21;
                                    auto tmp27 = static_cast<long>(192);
                                    auto tmp28 = tmp18 < tmp27;
                                    auto tmp29 = [&]
                                    {
                                        auto tmp30 = in_ptr1[static_cast<long>(384L + x2 + (576L*x1) + (451584L*x0))];
                                        return tmp30;
                                    }
                                    ;
                                    auto tmp31 = tmp26 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                                    auto tmp32 = tmp22 ? tmp25 : tmp31;
                                    return tmp32;
                                }
                                ;
                                auto tmp33 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                                auto tmp34 = tmp12 >= tmp15;
                                auto tmp35 = static_cast<long>(256);
                                auto tmp36 = tmp12 < tmp35;
                                auto tmp37 = [&]
                                {
                                    auto tmp38 = in_ptr2[static_cast<long>(320L + x2 + (576L*x1) + (451584L*x0))];
                                    return tmp38;
                                }
                                ;
                                auto tmp39 = tmp34 ? tmp37() : static_cast<decltype(tmp37())>(0.0);
                                auto tmp40 = tmp16 ? tmp33 : tmp39;
                                return tmp40;
                            }
                            ;
                            auto tmp41 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp42 = tmp6 >= tmp9;
                            auto tmp43 = static_cast<long>(320);
                            auto tmp44 = tmp6 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr3[static_cast<long>(256L + x2 + (576L*x1) + (451584L*x0))];
                                return tmp46;
                            }
                            ;
                            auto tmp47 = tmp42 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp48 = tmp10 ? tmp41 : tmp47;
                            return tmp48;
                        }
                        ;
                        auto tmp49 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp50 = tmp0 >= tmp3;
                        auto tmp51 = static_cast<long>(384);
                        auto tmp52 = tmp0 < tmp51;
                        auto tmp53 = [&]
                        {
                            auto tmp54 = in_ptr4[static_cast<long>(192L + x2 + (576L*x1) + (451584L*x0))];
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp50 ? tmp53() : static_cast<decltype(tmp53())>(0.0);
                        auto tmp56 = tmp4 ? tmp49 : tmp55;
                        out_ptr1[static_cast<long>(x1 + (784L*x2) + (702464L*x0))] = tmp56;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (702464L*x0)));
                            auto tmp1 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp13 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr9[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(0.001);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = std::sqrt(tmp6);
                            auto tmp8 = 1 / tmp7;
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp3 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                            tmp19.store(tmp20 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp20, 8, out_ptr2 + static_cast<long>(x1 + (896L*x2) + (702464L*x0)), static_cast<long>(896L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_26 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp31 = in_ptr3[static_cast<long>(x1)];
                        auto tmp33 = in_ptr4[static_cast<long>(x1)];
                        auto tmp41 = in_ptr5[static_cast<long>(x1)];
                        auto tmp43 = in_ptr6[static_cast<long>(x1)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(512);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (784L*x1) + (702464L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x1 + (576L*x2) + (451584L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(960);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = c10::convert<long>((-512L) + x1);
                            auto tmp15 = static_cast<long>(0);
                            auto tmp16 = tmp14 >= tmp15;
                            auto tmp17 = static_cast<long>(384);
                            auto tmp18 = tmp14 < tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = in_ptr2[static_cast<long>((-401408L) + x2 + (784L*x1) + (702464L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                            auto tmp22 = tmp14 >= tmp17;
                            auto tmp23 = static_cast<long>(448);
                            auto tmp24 = tmp14 < tmp23;
                            auto tmp25 = [&]
                            {
                                auto tmp26 = in_ptr1[static_cast<long>((-384L) + x1 + (576L*x2) + (451584L*x0))];
                                return tmp26;
                            }
                            ;
                            auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                            auto tmp28 = tmp18 ? tmp21 : tmp27;
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp10 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp30 = tmp4 ? tmp9 : tmp29;
                        auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                        auto tmp34 = static_cast<float>(0.001);
                        auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                        auto tmp36 = std::sqrt(tmp35);
                        auto tmp37 = 1 / tmp36;
                        auto tmp38 = static_cast<float>(1.0);
                        auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                        auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                        auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                        auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                        auto tmp45 = tmp44 * (tmp44>0);
                        out_ptr0[static_cast<long>(x1 + (960L*x2) + (752640L*x0))] = tmp45;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_29 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_31 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp47 = in_ptr4[static_cast<long>(x2)];
                        auto tmp49 = in_ptr5[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(512);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x1 + (784L*x2) + (702464L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (576L*x1) + (451584L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2 + (576L*x1) + (451584L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<long>(1024);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = c10::convert<long>((-512L) + x2);
                            auto tmp17 = static_cast<long>(0);
                            auto tmp18 = tmp16 >= tmp17;
                            auto tmp19 = static_cast<long>(448);
                            auto tmp20 = tmp16 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = c10::convert<long>((-512L) + x2);
                                auto tmp23 = static_cast<long>(0);
                                auto tmp24 = tmp22 >= tmp23;
                                auto tmp25 = static_cast<long>(384);
                                auto tmp26 = tmp22 < tmp25;
                                auto tmp27 = [&]
                                {
                                    auto tmp28 = in_ptr3[static_cast<long>((-401408L) + x1 + (784L*x2) + (702464L*x0))];
                                    return tmp28;
                                }
                                ;
                                auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                                auto tmp30 = tmp22 >= tmp25;
                                auto tmp31 = static_cast<long>(448);
                                auto tmp32 = tmp22 < tmp31;
                                auto tmp33 = [&]
                                {
                                    auto tmp34 = in_ptr1[static_cast<long>((-384L) + x2 + (576L*x1) + (451584L*x0))];
                                    return tmp34;
                                }
                                ;
                                auto tmp35 = tmp30 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                                auto tmp36 = tmp26 ? tmp29 : tmp35;
                                return tmp36;
                            }
                            ;
                            auto tmp37 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp38 = tmp16 >= tmp19;
                            auto tmp39 = static_cast<long>(512);
                            auto tmp40 = tmp16 < tmp39;
                            auto tmp41 = [&]
                            {
                                auto tmp42 = in_ptr2[static_cast<long>((-448L) + x2 + (576L*x1) + (451584L*x0))];
                                return tmp42;
                            }
                            ;
                            auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                            auto tmp44 = tmp20 ? tmp37 : tmp43;
                            return tmp44;
                        }
                        ;
                        auto tmp45 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp46 = tmp4 ? tmp11 : tmp45;
                        auto tmp48 = decltype(tmp46)(tmp46 - tmp47);
                        auto tmp50 = static_cast<float>(0.001);
                        auto tmp51 = decltype(tmp49)(tmp49 + tmp50);
                        auto tmp52 = std::sqrt(tmp51);
                        auto tmp53 = 1 / tmp52;
                        auto tmp54 = static_cast<float>(1.0);
                        auto tmp55 = decltype(tmp53)(tmp53 * tmp54);
                        auto tmp56 = decltype(tmp48)(tmp48 * tmp55);
                        out_ptr0[static_cast<long>(x2 + (1024L*x1) + (802816L*x0))] = tmp56;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = at::vec::clamp_min(tmp4, decltype(tmp4)(0));
                    tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_34 = async_compile.cpp('''
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
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(512);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x1 + (784L*x2) + (702464L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (576L*x1) + (451584L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2 + (576L*x1) + (451584L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            auto tmp11 = in_ptr3[static_cast<long>(x2 + (576L*x1) + (451584L*x0))];
                            auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp14 = tmp0 >= tmp3;
                        auto tmp15 = static_cast<long>(1088);
                        auto tmp16 = tmp0 < tmp15;
                        auto tmp17 = [&]
                        {
                            auto tmp18 = c10::convert<long>((-512L) + x2);
                            auto tmp19 = static_cast<long>(0);
                            auto tmp20 = tmp18 >= tmp19;
                            auto tmp21 = static_cast<long>(512);
                            auto tmp22 = tmp18 < tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>((-512L) + x2);
                                auto tmp25 = static_cast<long>(0);
                                auto tmp26 = tmp24 >= tmp25;
                                auto tmp27 = static_cast<long>(448);
                                auto tmp28 = tmp24 < tmp27;
                                auto tmp29 = [&]
                                {
                                    auto tmp30 = c10::convert<long>((-512L) + x2);
                                    auto tmp31 = static_cast<long>(0);
                                    auto tmp32 = tmp30 >= tmp31;
                                    auto tmp33 = static_cast<long>(384);
                                    auto tmp34 = tmp30 < tmp33;
                                    auto tmp35 = [&]
                                    {
                                        auto tmp36 = in_ptr4[static_cast<long>((-401408L) + x1 + (784L*x2) + (702464L*x0))];
                                        return tmp36;
                                    }
                                    ;
                                    auto tmp37 = tmp34 ? tmp35() : static_cast<decltype(tmp35())>(0.0);
                                    auto tmp38 = tmp30 >= tmp33;
                                    auto tmp39 = static_cast<long>(448);
                                    auto tmp40 = tmp30 < tmp39;
                                    auto tmp41 = [&]
                                    {
                                        auto tmp42 = in_ptr1[static_cast<long>((-384L) + x2 + (576L*x1) + (451584L*x0))];
                                        return tmp42;
                                    }
                                    ;
                                    auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                                    auto tmp44 = tmp34 ? tmp37 : tmp43;
                                    return tmp44;
                                }
                                ;
                                auto tmp45 = tmp28 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                                auto tmp46 = tmp24 >= tmp27;
                                auto tmp47 = static_cast<long>(512);
                                auto tmp48 = tmp24 < tmp47;
                                auto tmp49 = [&]
                                {
                                    auto tmp50 = in_ptr2[static_cast<long>((-448L) + x2 + (576L*x1) + (451584L*x0))];
                                    return tmp50;
                                }
                                ;
                                auto tmp51 = tmp46 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                auto tmp52 = tmp28 ? tmp45 : tmp51;
                                return tmp52;
                            }
                            ;
                            auto tmp53 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp54 = tmp18 >= tmp21;
                            auto tmp55 = static_cast<long>(576);
                            auto tmp56 = tmp18 < tmp55;
                            auto tmp57 = [&]
                            {
                                auto tmp58 = in_ptr3[static_cast<long>((-512L) + x2 + (576L*x1) + (451584L*x0))];
                                return tmp58;
                            }
                            ;
                            auto tmp59 = tmp54 ? tmp57() : static_cast<decltype(tmp57())>(0.0);
                            auto tmp60 = tmp22 ? tmp53 : tmp59;
                            return tmp60;
                        }
                        ;
                        auto tmp61 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                        auto tmp62 = tmp4 ? tmp13 : tmp61;
                        out_ptr0[static_cast<long>(x2 + (1088L*x1) + (852992L*x0))] = tmp62;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1088L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1088L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1088L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_35 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_37 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(640L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(576);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = c10::convert<long>(x2);
                            auto tmp7 = static_cast<long>(0);
                            auto tmp8 = tmp6 >= tmp7;
                            auto tmp9 = static_cast<long>(512);
                            auto tmp10 = tmp6 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(x2);
                                auto tmp13 = static_cast<long>(0);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = static_cast<long>(448);
                                auto tmp16 = tmp12 < tmp15;
                                auto tmp17 = [&]
                                {
                                    auto tmp18 = c10::convert<long>(x2);
                                    auto tmp19 = static_cast<long>(0);
                                    auto tmp20 = tmp18 >= tmp19;
                                    auto tmp21 = static_cast<long>(384);
                                    auto tmp22 = tmp18 < tmp21;
                                    auto tmp23 = [&]
                                    {
                                        auto tmp24 = in_ptr0[static_cast<long>(x1 + (784L*x2) + (702464L*x0))];
                                        return tmp24;
                                    }
                                    ;
                                    auto tmp25 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                                    auto tmp26 = tmp18 >= tmp21;
                                    auto tmp27 = static_cast<long>(448);
                                    auto tmp28 = tmp18 < tmp27;
                                    auto tmp29 = [&]
                                    {
                                        auto tmp30 = in_ptr1[static_cast<long>(128L + x2 + (576L*x1) + (451584L*x0))];
                                        return tmp30;
                                    }
                                    ;
                                    auto tmp31 = tmp26 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                                    auto tmp32 = tmp22 ? tmp25 : tmp31;
                                    return tmp32;
                                }
                                ;
                                auto tmp33 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                                auto tmp34 = tmp12 >= tmp15;
                                auto tmp35 = static_cast<long>(512);
                                auto tmp36 = tmp12 < tmp35;
                                auto tmp37 = [&]
                                {
                                    auto tmp38 = in_ptr2[static_cast<long>(64L + x2 + (576L*x1) + (451584L*x0))];
                                    return tmp38;
                                }
                                ;
                                auto tmp39 = tmp34 ? tmp37() : static_cast<decltype(tmp37())>(0.0);
                                auto tmp40 = tmp16 ? tmp33 : tmp39;
                                return tmp40;
                            }
                            ;
                            auto tmp41 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp42 = tmp6 >= tmp9;
                            auto tmp43 = static_cast<long>(576);
                            auto tmp44 = tmp6 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr3[static_cast<long>(x2 + (576L*x1) + (451584L*x0))];
                                return tmp46;
                            }
                            ;
                            auto tmp47 = tmp42 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp48 = tmp10 ? tmp41 : tmp47;
                            return tmp48;
                        }
                        ;
                        auto tmp49 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp50 = tmp0 >= tmp3;
                        auto tmp51 = static_cast<long>(640);
                        auto tmp52 = tmp0 < tmp51;
                        auto tmp53 = [&]
                        {
                            auto tmp54 = in_ptr4[static_cast<long>((-64L) + x2 + (576L*x1) + (451584L*x0))];
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp50 ? tmp53() : static_cast<decltype(tmp53())>(0.0);
                        auto tmp56 = tmp4 ? tmp49 : tmp55;
                        out_ptr0[static_cast<long>(x2 + (640L*x1) + (501760L*x0))] = tmp56;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(512);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr5[static_cast<long>(x1 + (784L*x2) + (702464L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (576L*x1) + (451584L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2 + (576L*x1) + (451584L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            auto tmp11 = in_ptr3[static_cast<long>(x2 + (576L*x1) + (451584L*x0))];
                            auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                            auto tmp13 = in_ptr4[static_cast<long>(x2 + (576L*x1) + (451584L*x0))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp16 = tmp0 >= tmp3;
                        auto tmp17 = static_cast<long>(1152);
                        auto tmp18 = tmp0 < tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = out_ptr0[static_cast<long>((-512L) + x2 + (640L*x1) + (501760L*x0))];
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp16 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp22 = tmp4 ? tmp15 : tmp21;
                        out_ptr1[static_cast<long>(x2 + (1152L*x1) + (903168L*x0))] = tmp22;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    auto tmp19 = tmp0 - tmp18;
                    auto tmp21 = tmp20 + tmp5;
                    auto tmp22 = tmp21.sqrt();
                    auto tmp23 = tmp22.reciprocal();
                    auto tmp24 = tmp23 * tmp10;
                    auto tmp25 = tmp19 * tmp24;
                    auto tmp27 = tmp25 * tmp26;
                    auto tmp29 = tmp27 + tmp28;
                    auto tmp30 = at::vec::clamp_min(tmp29, decltype(tmp29)(0));
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1152L*x0)));
                    tmp30.store(out_ptr3 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_38 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1216L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr2[static_cast<long>(x1)];
                    auto tmp33 = in_ptr3[static_cast<long>(x1)];
                    auto tmp41 = in_ptr4[static_cast<long>(x1)];
                    auto tmp43 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(1024);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (1152L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (1088L*x0))];
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        return tmp8;
                    }
                    ;
                    auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp10 = tmp0 >= tmp3;
                    auto tmp11 = static_cast<long>(1216);
                    auto tmp12 = tmp0 < tmp11;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = c10::convert<long>((-1024L) + x1);
                        auto tmp15 = static_cast<long>(0);
                        auto tmp16 = tmp14 >= tmp15;
                        auto tmp17 = static_cast<long>(128);
                        auto tmp18 = tmp14 < tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = in_ptr0[static_cast<long>(x1 + (1152L*x0))];
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp22 = tmp14 >= tmp17;
                        auto tmp23 = static_cast<long>(192);
                        auto tmp24 = tmp14 < tmp23;
                        auto tmp25 = [&]
                        {
                            auto tmp26 = in_ptr1[static_cast<long>((-128L) + x1 + (1088L*x0))];
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                        auto tmp28 = tmp18 ? tmp21 : tmp27;
                        return tmp28;
                    }
                    ;
                    auto tmp29 = tmp10 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp30 = tmp4 ? tmp9 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(0.001);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    auto tmp45 = tmp44 * (tmp44>0);
                    out_ptr0[static_cast<long>(x1 + (1216L*x0))] = tmp45;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_41 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1280L); x1+=static_cast<long>(1L))
                {
                    auto tmp47 = in_ptr3[static_cast<long>(x1)];
                    auto tmp49 = in_ptr4[static_cast<long>(x1)];
                    auto tmp57 = in_ptr5[static_cast<long>(x1)];
                    auto tmp59 = in_ptr6[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(1024);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (1152L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (1088L*x0))];
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = in_ptr2[static_cast<long>(x1 + (1088L*x0))];
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1280);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = c10::convert<long>((-1024L) + x1);
                        auto tmp17 = static_cast<long>(0);
                        auto tmp18 = tmp16 >= tmp17;
                        auto tmp19 = static_cast<long>(192);
                        auto tmp20 = tmp16 < tmp19;
                        auto tmp21 = [&]
                        {
                            auto tmp22 = c10::convert<long>((-1024L) + x1);
                            auto tmp23 = static_cast<long>(0);
                            auto tmp24 = tmp22 >= tmp23;
                            auto tmp25 = static_cast<long>(128);
                            auto tmp26 = tmp22 < tmp25;
                            auto tmp27 = [&]
                            {
                                auto tmp28 = in_ptr0[static_cast<long>(x1 + (1152L*x0))];
                                return tmp28;
                            }
                            ;
                            auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                            auto tmp30 = tmp22 >= tmp25;
                            auto tmp31 = static_cast<long>(192);
                            auto tmp32 = tmp22 < tmp31;
                            auto tmp33 = [&]
                            {
                                auto tmp34 = in_ptr1[static_cast<long>((-128L) + x1 + (1088L*x0))];
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp30 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                            auto tmp36 = tmp26 ? tmp29 : tmp35;
                            return tmp36;
                        }
                        ;
                        auto tmp37 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                        auto tmp38 = tmp16 >= tmp19;
                        auto tmp39 = static_cast<long>(256);
                        auto tmp40 = tmp16 < tmp39;
                        auto tmp41 = [&]
                        {
                            auto tmp42 = in_ptr2[static_cast<long>((-192L) + x1 + (1088L*x0))];
                            return tmp42;
                        }
                        ;
                        auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                        auto tmp44 = tmp20 ? tmp37 : tmp43;
                        return tmp44;
                    }
                    ;
                    auto tmp45 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp46 = tmp4 ? tmp11 : tmp45;
                    auto tmp48 = decltype(tmp46)(tmp46 - tmp47);
                    auto tmp50 = static_cast<float>(0.001);
                    auto tmp51 = decltype(tmp49)(tmp49 + tmp50);
                    auto tmp52 = std::sqrt(tmp51);
                    auto tmp53 = 1 / tmp52;
                    auto tmp54 = static_cast<float>(1.0);
                    auto tmp55 = decltype(tmp53)(tmp53 * tmp54);
                    auto tmp56 = decltype(tmp48)(tmp48 * tmp55);
                    auto tmp58 = decltype(tmp56)(tmp56 * tmp57);
                    auto tmp60 = decltype(tmp58)(tmp58 + tmp59);
                    auto tmp61 = tmp60 * (tmp60>0);
                    in_out_ptr0[static_cast<long>(x1 + (1280L*x0))] = tmp61;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_44 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_46 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1344L); x1+=static_cast<long>(1L))
                {
                    auto tmp63 = in_ptr4[static_cast<long>(x1)];
                    auto tmp65 = in_ptr5[static_cast<long>(x1)];
                    auto tmp73 = in_ptr6[static_cast<long>(x1)];
                    auto tmp75 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(1024);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (1152L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (1088L*x0))];
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = in_ptr2[static_cast<long>(x1 + (1088L*x0))];
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = in_ptr3[static_cast<long>(x1 + (1088L*x0))];
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp14 = tmp0 >= tmp3;
                    auto tmp15 = static_cast<long>(1344);
                    auto tmp16 = tmp0 < tmp15;
                    auto tmp17 = [&]
                    {
                        auto tmp18 = c10::convert<long>((-1024L) + x1);
                        auto tmp19 = static_cast<long>(0);
                        auto tmp20 = tmp18 >= tmp19;
                        auto tmp21 = static_cast<long>(256);
                        auto tmp22 = tmp18 < tmp21;
                        auto tmp23 = [&]
                        {
                            auto tmp24 = c10::convert<long>((-1024L) + x1);
                            auto tmp25 = static_cast<long>(0);
                            auto tmp26 = tmp24 >= tmp25;
                            auto tmp27 = static_cast<long>(192);
                            auto tmp28 = tmp24 < tmp27;
                            auto tmp29 = [&]
                            {
                                auto tmp30 = c10::convert<long>((-1024L) + x1);
                                auto tmp31 = static_cast<long>(0);
                                auto tmp32 = tmp30 >= tmp31;
                                auto tmp33 = static_cast<long>(128);
                                auto tmp34 = tmp30 < tmp33;
                                auto tmp35 = [&]
                                {
                                    auto tmp36 = in_ptr0[static_cast<long>(x1 + (1152L*x0))];
                                    return tmp36;
                                }
                                ;
                                auto tmp37 = tmp34 ? tmp35() : static_cast<decltype(tmp35())>(0.0);
                                auto tmp38 = tmp30 >= tmp33;
                                auto tmp39 = static_cast<long>(192);
                                auto tmp40 = tmp30 < tmp39;
                                auto tmp41 = [&]
                                {
                                    auto tmp42 = in_ptr1[static_cast<long>((-128L) + x1 + (1088L*x0))];
                                    return tmp42;
                                }
                                ;
                                auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                                auto tmp44 = tmp34 ? tmp37 : tmp43;
                                return tmp44;
                            }
                            ;
                            auto tmp45 = tmp28 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                            auto tmp46 = tmp24 >= tmp27;
                            auto tmp47 = static_cast<long>(256);
                            auto tmp48 = tmp24 < tmp47;
                            auto tmp49 = [&]
                            {
                                auto tmp50 = in_ptr2[static_cast<long>((-192L) + x1 + (1088L*x0))];
                                return tmp50;
                            }
                            ;
                            auto tmp51 = tmp46 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                            auto tmp52 = tmp28 ? tmp45 : tmp51;
                            return tmp52;
                        }
                        ;
                        auto tmp53 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                        auto tmp54 = tmp18 >= tmp21;
                        auto tmp55 = static_cast<long>(320);
                        auto tmp56 = tmp18 < tmp55;
                        auto tmp57 = [&]
                        {
                            auto tmp58 = in_ptr3[static_cast<long>((-256L) + x1 + (1088L*x0))];
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp54 ? tmp57() : static_cast<decltype(tmp57())>(0.0);
                        auto tmp60 = tmp22 ? tmp53 : tmp59;
                        return tmp60;
                    }
                    ;
                    auto tmp61 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                    auto tmp62 = tmp4 ? tmp13 : tmp61;
                    auto tmp64 = decltype(tmp62)(tmp62 - tmp63);
                    auto tmp66 = static_cast<float>(0.001);
                    auto tmp67 = decltype(tmp65)(tmp65 + tmp66);
                    auto tmp68 = std::sqrt(tmp67);
                    auto tmp69 = 1 / tmp68;
                    auto tmp70 = static_cast<float>(1.0);
                    auto tmp71 = decltype(tmp69)(tmp69 * tmp70);
                    auto tmp72 = decltype(tmp64)(tmp64 * tmp71);
                    auto tmp74 = decltype(tmp72)(tmp72 * tmp73);
                    auto tmp76 = decltype(tmp74)(tmp74 + tmp75);
                    auto tmp77 = tmp76 * (tmp76>0);
                    in_out_ptr0[static_cast<long>(x1 + (1344L*x0))] = tmp77;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_relu_49 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp9[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1152L*x1) + (1152L*x1_inner) + (225792L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1088L*x1) + (1088L*x1_inner) + (213248L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1088L*x1) + (1088L*x1_inner) + (213248L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1088L*x1) + (1088L*x1_inner) + (213248L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1088L*x1) + (1088L*x1_inner) + (213248L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (275968L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1152L*x1) + (225792L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                        auto tmp7 = in_ptr4[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (275968L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(320);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = c10::convert<long>(x2);
                            auto tmp7 = static_cast<long>(0);
                            auto tmp8 = tmp6 >= tmp7;
                            auto tmp9 = static_cast<long>(256);
                            auto tmp10 = tmp6 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(x2);
                                auto tmp13 = static_cast<long>(0);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = static_cast<long>(192);
                                auto tmp16 = tmp12 < tmp15;
                                auto tmp17 = [&]
                                {
                                    auto tmp18 = c10::convert<long>(x2);
                                    auto tmp19 = static_cast<long>(0);
                                    auto tmp20 = tmp18 >= tmp19;
                                    auto tmp21 = static_cast<long>(128);
                                    auto tmp22 = tmp18 < tmp21;
                                    auto tmp23 = [&]
                                    {
                                        auto tmp24 = in_ptr0[static_cast<long>(1024L + x2 + (1152L*x1) + (225792L*x0))];
                                        return tmp24;
                                    }
                                    ;
                                    auto tmp25 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                                    auto tmp26 = tmp18 >= tmp21;
                                    auto tmp27 = static_cast<long>(192);
                                    auto tmp28 = tmp18 < tmp27;
                                    auto tmp29 = [&]
                                    {
                                        auto tmp30 = in_ptr1[static_cast<long>(896L + x2 + (1088L*x1) + (213248L*x0))];
                                        return tmp30;
                                    }
                                    ;
                                    auto tmp31 = tmp26 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                                    auto tmp32 = tmp22 ? tmp25 : tmp31;
                                    return tmp32;
                                }
                                ;
                                auto tmp33 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                                auto tmp34 = tmp12 >= tmp15;
                                auto tmp35 = static_cast<long>(256);
                                auto tmp36 = tmp12 < tmp35;
                                auto tmp37 = [&]
                                {
                                    auto tmp38 = in_ptr2[static_cast<long>(832L + x2 + (1088L*x1) + (213248L*x0))];
                                    return tmp38;
                                }
                                ;
                                auto tmp39 = tmp34 ? tmp37() : static_cast<decltype(tmp37())>(0.0);
                                auto tmp40 = tmp16 ? tmp33 : tmp39;
                                return tmp40;
                            }
                            ;
                            auto tmp41 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp42 = tmp6 >= tmp9;
                            auto tmp43 = static_cast<long>(320);
                            auto tmp44 = tmp6 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr3[static_cast<long>(768L + x2 + (1088L*x1) + (213248L*x0))];
                                return tmp46;
                            }
                            ;
                            auto tmp47 = tmp42 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp48 = tmp10 ? tmp41 : tmp47;
                            return tmp48;
                        }
                        ;
                        auto tmp49 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp50 = tmp0 >= tmp3;
                        auto tmp51 = static_cast<long>(384);
                        auto tmp52 = tmp0 < tmp51;
                        auto tmp53 = [&]
                        {
                            auto tmp54 = in_ptr4[static_cast<long>(704L + x2 + (1088L*x1) + (213248L*x0))];
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp50 ? tmp53() : static_cast<decltype(tmp53())>(0.0);
                        auto tmp56 = tmp4 ? tmp49 : tmp55;
                        out_ptr1[static_cast<long>(x1 + (196L*x2) + (275968L*x0))] = tmp56;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1408L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (275968L*x0)));
                            auto tmp1 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp13 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr9[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(0.001);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = std::sqrt(tmp6);
                            auto tmp8 = 1 / tmp7;
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp3 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                            tmp19.store(tmp20 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp20, 8, out_ptr2 + static_cast<long>(x1 + (1408L*x2) + (275968L*x0)), static_cast<long>(1408L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (275968L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(0.001);
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
                        tmp17.store(out_ptr2 + static_cast<long>(x1 + (1408L*x2) + (275968L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_50 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1472L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp31 = in_ptr3[static_cast<long>(x1)];
                        auto tmp33 = in_ptr4[static_cast<long>(x1)];
                        auto tmp41 = in_ptr5[static_cast<long>(x1)];
                        auto tmp43 = in_ptr6[static_cast<long>(x1)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (275968L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x1 + (1088L*x2) + (213248L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(1472);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = c10::convert<long>((-1024L) + x1);
                            auto tmp15 = static_cast<long>(0);
                            auto tmp16 = tmp14 >= tmp15;
                            auto tmp17 = static_cast<long>(384);
                            auto tmp18 = tmp14 < tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = in_ptr2[static_cast<long>((-200704L) + x2 + (196L*x1) + (275968L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                            auto tmp22 = tmp14 >= tmp17;
                            auto tmp23 = static_cast<long>(448);
                            auto tmp24 = tmp14 < tmp23;
                            auto tmp25 = [&]
                            {
                                auto tmp26 = in_ptr1[static_cast<long>((-384L) + x1 + (1088L*x2) + (213248L*x0))];
                                return tmp26;
                            }
                            ;
                            auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                            auto tmp28 = tmp18 ? tmp21 : tmp27;
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp10 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp30 = tmp4 ? tmp9 : tmp29;
                        auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                        auto tmp34 = static_cast<float>(0.001);
                        auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                        auto tmp36 = std::sqrt(tmp35);
                        auto tmp37 = 1 / tmp36;
                        auto tmp38 = static_cast<float>(1.0);
                        auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                        auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                        auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                        auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                        auto tmp45 = tmp44 * (tmp44>0);
                        out_ptr0[static_cast<long>(x1 + (1472L*x2) + (288512L*x0))] = tmp45;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_53 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_55 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp47 = in_ptr4[static_cast<long>(x2)];
                        auto tmp49 = in_ptr5[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (275968L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<long>(1536);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = c10::convert<long>((-1024L) + x2);
                            auto tmp17 = static_cast<long>(0);
                            auto tmp18 = tmp16 >= tmp17;
                            auto tmp19 = static_cast<long>(448);
                            auto tmp20 = tmp16 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = c10::convert<long>((-1024L) + x2);
                                auto tmp23 = static_cast<long>(0);
                                auto tmp24 = tmp22 >= tmp23;
                                auto tmp25 = static_cast<long>(384);
                                auto tmp26 = tmp22 < tmp25;
                                auto tmp27 = [&]
                                {
                                    auto tmp28 = in_ptr3[static_cast<long>((-200704L) + x1 + (196L*x2) + (275968L*x0))];
                                    return tmp28;
                                }
                                ;
                                auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                                auto tmp30 = tmp22 >= tmp25;
                                auto tmp31 = static_cast<long>(448);
                                auto tmp32 = tmp22 < tmp31;
                                auto tmp33 = [&]
                                {
                                    auto tmp34 = in_ptr1[static_cast<long>((-384L) + x2 + (1088L*x1) + (213248L*x0))];
                                    return tmp34;
                                }
                                ;
                                auto tmp35 = tmp30 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                                auto tmp36 = tmp26 ? tmp29 : tmp35;
                                return tmp36;
                            }
                            ;
                            auto tmp37 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp38 = tmp16 >= tmp19;
                            auto tmp39 = static_cast<long>(512);
                            auto tmp40 = tmp16 < tmp39;
                            auto tmp41 = [&]
                            {
                                auto tmp42 = in_ptr2[static_cast<long>((-448L) + x2 + (1088L*x1) + (213248L*x0))];
                                return tmp42;
                            }
                            ;
                            auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                            auto tmp44 = tmp20 ? tmp37 : tmp43;
                            return tmp44;
                        }
                        ;
                        auto tmp45 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp46 = tmp4 ? tmp11 : tmp45;
                        auto tmp48 = decltype(tmp46)(tmp46 - tmp47);
                        auto tmp50 = static_cast<float>(0.001);
                        auto tmp51 = decltype(tmp49)(tmp49 + tmp50);
                        auto tmp52 = std::sqrt(tmp51);
                        auto tmp53 = 1 / tmp52;
                        auto tmp54 = static_cast<float>(1.0);
                        auto tmp55 = decltype(tmp53)(tmp53 * tmp54);
                        auto tmp56 = decltype(tmp48)(tmp48 * tmp55);
                        out_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp56;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = at::vec::clamp_min(tmp4, decltype(tmp4)(0));
                    tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_56 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_58 = async_compile.cpp('''
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
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1600L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (275968L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            auto tmp11 = in_ptr3[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp14 = tmp0 >= tmp3;
                        auto tmp15 = static_cast<long>(1600);
                        auto tmp16 = tmp0 < tmp15;
                        auto tmp17 = [&]
                        {
                            auto tmp18 = c10::convert<long>((-1024L) + x2);
                            auto tmp19 = static_cast<long>(0);
                            auto tmp20 = tmp18 >= tmp19;
                            auto tmp21 = static_cast<long>(512);
                            auto tmp22 = tmp18 < tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>((-1024L) + x2);
                                auto tmp25 = static_cast<long>(0);
                                auto tmp26 = tmp24 >= tmp25;
                                auto tmp27 = static_cast<long>(448);
                                auto tmp28 = tmp24 < tmp27;
                                auto tmp29 = [&]
                                {
                                    auto tmp30 = c10::convert<long>((-1024L) + x2);
                                    auto tmp31 = static_cast<long>(0);
                                    auto tmp32 = tmp30 >= tmp31;
                                    auto tmp33 = static_cast<long>(384);
                                    auto tmp34 = tmp30 < tmp33;
                                    auto tmp35 = [&]
                                    {
                                        auto tmp36 = in_ptr4[static_cast<long>((-200704L) + x1 + (196L*x2) + (275968L*x0))];
                                        return tmp36;
                                    }
                                    ;
                                    auto tmp37 = tmp34 ? tmp35() : static_cast<decltype(tmp35())>(0.0);
                                    auto tmp38 = tmp30 >= tmp33;
                                    auto tmp39 = static_cast<long>(448);
                                    auto tmp40 = tmp30 < tmp39;
                                    auto tmp41 = [&]
                                    {
                                        auto tmp42 = in_ptr1[static_cast<long>((-384L) + x2 + (1088L*x1) + (213248L*x0))];
                                        return tmp42;
                                    }
                                    ;
                                    auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                                    auto tmp44 = tmp34 ? tmp37 : tmp43;
                                    return tmp44;
                                }
                                ;
                                auto tmp45 = tmp28 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                                auto tmp46 = tmp24 >= tmp27;
                                auto tmp47 = static_cast<long>(512);
                                auto tmp48 = tmp24 < tmp47;
                                auto tmp49 = [&]
                                {
                                    auto tmp50 = in_ptr2[static_cast<long>((-448L) + x2 + (1088L*x1) + (213248L*x0))];
                                    return tmp50;
                                }
                                ;
                                auto tmp51 = tmp46 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                auto tmp52 = tmp28 ? tmp45 : tmp51;
                                return tmp52;
                            }
                            ;
                            auto tmp53 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp54 = tmp18 >= tmp21;
                            auto tmp55 = static_cast<long>(576);
                            auto tmp56 = tmp18 < tmp55;
                            auto tmp57 = [&]
                            {
                                auto tmp58 = in_ptr3[static_cast<long>((-512L) + x2 + (1088L*x1) + (213248L*x0))];
                                return tmp58;
                            }
                            ;
                            auto tmp59 = tmp54 ? tmp57() : static_cast<decltype(tmp57())>(0.0);
                            auto tmp60 = tmp22 ? tmp53 : tmp59;
                            return tmp60;
                        }
                        ;
                        auto tmp61 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                        auto tmp62 = tmp4 ? tmp13 : tmp61;
                        out_ptr0[static_cast<long>(x2 + (1600L*x1) + (313600L*x0))] = tmp62;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1600L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_59 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_relu_61 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (275968L*x0)), static_cast<long>(196L), tmp0, 8);
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1088L*x1) + (1088L*x1_inner) + (213248L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1088L*x1) + (1088L*x1_inner) + (213248L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1088L*x1) + (1088L*x1_inner) + (213248L*x0)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1088L*x1) + (1088L*x1_inner) + (213248L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = tmp5 + tmp6;
                            auto tmp9 = tmp7 + tmp8;
                            tmp9.store(tmp10 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp10, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (326144L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (275968L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                        auto tmp7 = in_ptr4[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (326144L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(640L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(576);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = c10::convert<long>(x2);
                            auto tmp7 = static_cast<long>(0);
                            auto tmp8 = tmp6 >= tmp7;
                            auto tmp9 = static_cast<long>(512);
                            auto tmp10 = tmp6 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(x2);
                                auto tmp13 = static_cast<long>(0);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = static_cast<long>(448);
                                auto tmp16 = tmp12 < tmp15;
                                auto tmp17 = [&]
                                {
                                    auto tmp18 = c10::convert<long>(x2);
                                    auto tmp19 = static_cast<long>(0);
                                    auto tmp20 = tmp18 >= tmp19;
                                    auto tmp21 = static_cast<long>(384);
                                    auto tmp22 = tmp18 < tmp21;
                                    auto tmp23 = [&]
                                    {
                                        auto tmp24 = in_ptr5[static_cast<long>(x1 + (196L*x2) + (275968L*x0))];
                                        return tmp24;
                                    }
                                    ;
                                    auto tmp25 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                                    auto tmp26 = tmp18 >= tmp21;
                                    auto tmp27 = static_cast<long>(448);
                                    auto tmp28 = tmp18 < tmp27;
                                    auto tmp29 = [&]
                                    {
                                        auto tmp30 = in_ptr1[static_cast<long>(640L + x2 + (1088L*x1) + (213248L*x0))];
                                        return tmp30;
                                    }
                                    ;
                                    auto tmp31 = tmp26 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                                    auto tmp32 = tmp22 ? tmp25 : tmp31;
                                    return tmp32;
                                }
                                ;
                                auto tmp33 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                                auto tmp34 = tmp12 >= tmp15;
                                auto tmp35 = static_cast<long>(512);
                                auto tmp36 = tmp12 < tmp35;
                                auto tmp37 = [&]
                                {
                                    auto tmp38 = in_ptr2[static_cast<long>(576L + x2 + (1088L*x1) + (213248L*x0))];
                                    return tmp38;
                                }
                                ;
                                auto tmp39 = tmp34 ? tmp37() : static_cast<decltype(tmp37())>(0.0);
                                auto tmp40 = tmp16 ? tmp33 : tmp39;
                                return tmp40;
                            }
                            ;
                            auto tmp41 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp42 = tmp6 >= tmp9;
                            auto tmp43 = static_cast<long>(576);
                            auto tmp44 = tmp6 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr3[static_cast<long>(512L + x2 + (1088L*x1) + (213248L*x0))];
                                return tmp46;
                            }
                            ;
                            auto tmp47 = tmp42 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp48 = tmp10 ? tmp41 : tmp47;
                            return tmp48;
                        }
                        ;
                        auto tmp49 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp50 = tmp0 >= tmp3;
                        auto tmp51 = static_cast<long>(640);
                        auto tmp52 = tmp0 < tmp51;
                        auto tmp53 = [&]
                        {
                            auto tmp54 = in_ptr4[static_cast<long>(448L + x2 + (1088L*x1) + (213248L*x0))];
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp50 ? tmp53() : static_cast<decltype(tmp53())>(0.0);
                        auto tmp56 = tmp4 ? tmp49 : tmp55;
                        out_ptr1[static_cast<long>(x1 + (196L*x2) + (326144L*x0))] = tmp56;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1664L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (326144L*x0)));
                            auto tmp1 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp13 = in_ptr9[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr10[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(0.001);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = std::sqrt(tmp6);
                            auto tmp8 = 1 / tmp7;
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp3 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                            tmp19.store(tmp20 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp20, 8, out_ptr2 + static_cast<long>(x1 + (1664L*x2) + (326144L*x0)), static_cast<long>(1664L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (326144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(0.001);
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
                        tmp17.store(out_ptr2 + static_cast<long>(x1 + (1664L*x2) + (326144L*x0)));
                    }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1728L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp31 = in_ptr3[static_cast<long>(x1)];
                        auto tmp33 = in_ptr4[static_cast<long>(x1)];
                        auto tmp41 = in_ptr5[static_cast<long>(x1)];
                        auto tmp43 = in_ptr6[static_cast<long>(x1)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (326144L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x1 + (1088L*x2) + (213248L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(1728);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = c10::convert<long>((-1024L) + x1);
                            auto tmp15 = static_cast<long>(0);
                            auto tmp16 = tmp14 >= tmp15;
                            auto tmp17 = static_cast<long>(640);
                            auto tmp18 = tmp14 < tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = in_ptr2[static_cast<long>((-200704L) + x2 + (196L*x1) + (326144L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                            auto tmp22 = tmp14 >= tmp17;
                            auto tmp23 = static_cast<long>(704);
                            auto tmp24 = tmp14 < tmp23;
                            auto tmp25 = [&]
                            {
                                auto tmp26 = in_ptr1[static_cast<long>((-640L) + x1 + (1088L*x2) + (213248L*x0))];
                                return tmp26;
                            }
                            ;
                            auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                            auto tmp28 = tmp18 ? tmp21 : tmp27;
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp10 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp30 = tmp4 ? tmp9 : tmp29;
                        auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                        auto tmp34 = static_cast<float>(0.001);
                        auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                        auto tmp36 = std::sqrt(tmp35);
                        auto tmp37 = 1 / tmp36;
                        auto tmp38 = static_cast<float>(1.0);
                        auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                        auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                        auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                        auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                        auto tmp45 = tmp44 * (tmp44>0);
                        out_ptr0[static_cast<long>(x1 + (1728L*x2) + (338688L*x0))] = tmp45;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_65 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_67 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1792L); x2+=static_cast<long>(1L))
                    {
                        auto tmp47 = in_ptr4[static_cast<long>(x2)];
                        auto tmp49 = in_ptr5[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (326144L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<long>(1792);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = c10::convert<long>((-1024L) + x2);
                            auto tmp17 = static_cast<long>(0);
                            auto tmp18 = tmp16 >= tmp17;
                            auto tmp19 = static_cast<long>(704);
                            auto tmp20 = tmp16 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = c10::convert<long>((-1024L) + x2);
                                auto tmp23 = static_cast<long>(0);
                                auto tmp24 = tmp22 >= tmp23;
                                auto tmp25 = static_cast<long>(640);
                                auto tmp26 = tmp22 < tmp25;
                                auto tmp27 = [&]
                                {
                                    auto tmp28 = in_ptr3[static_cast<long>((-200704L) + x1 + (196L*x2) + (326144L*x0))];
                                    return tmp28;
                                }
                                ;
                                auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                                auto tmp30 = tmp22 >= tmp25;
                                auto tmp31 = static_cast<long>(704);
                                auto tmp32 = tmp22 < tmp31;
                                auto tmp33 = [&]
                                {
                                    auto tmp34 = in_ptr1[static_cast<long>((-640L) + x2 + (1088L*x1) + (213248L*x0))];
                                    return tmp34;
                                }
                                ;
                                auto tmp35 = tmp30 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                                auto tmp36 = tmp26 ? tmp29 : tmp35;
                                return tmp36;
                            }
                            ;
                            auto tmp37 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp38 = tmp16 >= tmp19;
                            auto tmp39 = static_cast<long>(768);
                            auto tmp40 = tmp16 < tmp39;
                            auto tmp41 = [&]
                            {
                                auto tmp42 = in_ptr2[static_cast<long>((-704L) + x2 + (1088L*x1) + (213248L*x0))];
                                return tmp42;
                            }
                            ;
                            auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                            auto tmp44 = tmp20 ? tmp37 : tmp43;
                            return tmp44;
                        }
                        ;
                        auto tmp45 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp46 = tmp4 ? tmp11 : tmp45;
                        auto tmp48 = decltype(tmp46)(tmp46 - tmp47);
                        auto tmp50 = static_cast<float>(0.001);
                        auto tmp51 = decltype(tmp49)(tmp49 + tmp50);
                        auto tmp52 = std::sqrt(tmp51);
                        auto tmp53 = 1 / tmp52;
                        auto tmp54 = static_cast<float>(1.0);
                        auto tmp55 = decltype(tmp53)(tmp53 * tmp54);
                        auto tmp56 = decltype(tmp48)(tmp48 * tmp55);
                        out_ptr0[static_cast<long>(x2 + (1792L*x1) + (351232L*x0))] = tmp56;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1792L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1792L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = at::vec::clamp_min(tmp4, decltype(tmp4)(0));
                    tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (1792L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_68 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_70 = async_compile.cpp('''
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
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1856L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (326144L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            auto tmp11 = in_ptr3[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp14 = tmp0 >= tmp3;
                        auto tmp15 = static_cast<long>(1856);
                        auto tmp16 = tmp0 < tmp15;
                        auto tmp17 = [&]
                        {
                            auto tmp18 = c10::convert<long>((-1024L) + x2);
                            auto tmp19 = static_cast<long>(0);
                            auto tmp20 = tmp18 >= tmp19;
                            auto tmp21 = static_cast<long>(768);
                            auto tmp22 = tmp18 < tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>((-1024L) + x2);
                                auto tmp25 = static_cast<long>(0);
                                auto tmp26 = tmp24 >= tmp25;
                                auto tmp27 = static_cast<long>(704);
                                auto tmp28 = tmp24 < tmp27;
                                auto tmp29 = [&]
                                {
                                    auto tmp30 = c10::convert<long>((-1024L) + x2);
                                    auto tmp31 = static_cast<long>(0);
                                    auto tmp32 = tmp30 >= tmp31;
                                    auto tmp33 = static_cast<long>(640);
                                    auto tmp34 = tmp30 < tmp33;
                                    auto tmp35 = [&]
                                    {
                                        auto tmp36 = in_ptr4[static_cast<long>((-200704L) + x1 + (196L*x2) + (326144L*x0))];
                                        return tmp36;
                                    }
                                    ;
                                    auto tmp37 = tmp34 ? tmp35() : static_cast<decltype(tmp35())>(0.0);
                                    auto tmp38 = tmp30 >= tmp33;
                                    auto tmp39 = static_cast<long>(704);
                                    auto tmp40 = tmp30 < tmp39;
                                    auto tmp41 = [&]
                                    {
                                        auto tmp42 = in_ptr1[static_cast<long>((-640L) + x2 + (1088L*x1) + (213248L*x0))];
                                        return tmp42;
                                    }
                                    ;
                                    auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                                    auto tmp44 = tmp34 ? tmp37 : tmp43;
                                    return tmp44;
                                }
                                ;
                                auto tmp45 = tmp28 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                                auto tmp46 = tmp24 >= tmp27;
                                auto tmp47 = static_cast<long>(768);
                                auto tmp48 = tmp24 < tmp47;
                                auto tmp49 = [&]
                                {
                                    auto tmp50 = in_ptr2[static_cast<long>((-704L) + x2 + (1088L*x1) + (213248L*x0))];
                                    return tmp50;
                                }
                                ;
                                auto tmp51 = tmp46 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                auto tmp52 = tmp28 ? tmp45 : tmp51;
                                return tmp52;
                            }
                            ;
                            auto tmp53 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp54 = tmp18 >= tmp21;
                            auto tmp55 = static_cast<long>(832);
                            auto tmp56 = tmp18 < tmp55;
                            auto tmp57 = [&]
                            {
                                auto tmp58 = in_ptr3[static_cast<long>((-768L) + x2 + (1088L*x1) + (213248L*x0))];
                                return tmp58;
                            }
                            ;
                            auto tmp59 = tmp54 ? tmp57() : static_cast<decltype(tmp57())>(0.0);
                            auto tmp60 = tmp22 ? tmp53 : tmp59;
                            return tmp60;
                        }
                        ;
                        auto tmp61 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                        auto tmp62 = tmp4 ? tmp13 : tmp61;
                        out_ptr0[static_cast<long>(x2 + (1856L*x1) + (363776L*x0))] = tmp62;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1856L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1856L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1856L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_71 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_relu_73 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (326144L*x0)), static_cast<long>(196L), tmp0, 8);
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1088L*x1) + (1088L*x1_inner) + (213248L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1088L*x1) + (1088L*x1_inner) + (213248L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1088L*x1) + (1088L*x1_inner) + (213248L*x0)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1088L*x1) + (1088L*x1_inner) + (213248L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = tmp5 + tmp6;
                            auto tmp9 = tmp7 + tmp8;
                            tmp9.store(tmp10 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp10, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (376320L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (326144L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                        auto tmp7 = in_ptr4[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (376320L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(832);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = c10::convert<long>(x2);
                            auto tmp7 = static_cast<long>(0);
                            auto tmp8 = tmp6 >= tmp7;
                            auto tmp9 = static_cast<long>(768);
                            auto tmp10 = tmp6 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(x2);
                                auto tmp13 = static_cast<long>(0);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = static_cast<long>(704);
                                auto tmp16 = tmp12 < tmp15;
                                auto tmp17 = [&]
                                {
                                    auto tmp18 = c10::convert<long>(x2);
                                    auto tmp19 = static_cast<long>(0);
                                    auto tmp20 = tmp18 >= tmp19;
                                    auto tmp21 = static_cast<long>(640);
                                    auto tmp22 = tmp18 < tmp21;
                                    auto tmp23 = [&]
                                    {
                                        auto tmp24 = in_ptr5[static_cast<long>(x1 + (196L*x2) + (326144L*x0))];
                                        return tmp24;
                                    }
                                    ;
                                    auto tmp25 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                                    auto tmp26 = tmp18 >= tmp21;
                                    auto tmp27 = static_cast<long>(704);
                                    auto tmp28 = tmp18 < tmp27;
                                    auto tmp29 = [&]
                                    {
                                        auto tmp30 = in_ptr1[static_cast<long>(384L + x2 + (1088L*x1) + (213248L*x0))];
                                        return tmp30;
                                    }
                                    ;
                                    auto tmp31 = tmp26 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                                    auto tmp32 = tmp22 ? tmp25 : tmp31;
                                    return tmp32;
                                }
                                ;
                                auto tmp33 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                                auto tmp34 = tmp12 >= tmp15;
                                auto tmp35 = static_cast<long>(768);
                                auto tmp36 = tmp12 < tmp35;
                                auto tmp37 = [&]
                                {
                                    auto tmp38 = in_ptr2[static_cast<long>(320L + x2 + (1088L*x1) + (213248L*x0))];
                                    return tmp38;
                                }
                                ;
                                auto tmp39 = tmp34 ? tmp37() : static_cast<decltype(tmp37())>(0.0);
                                auto tmp40 = tmp16 ? tmp33 : tmp39;
                                return tmp40;
                            }
                            ;
                            auto tmp41 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp42 = tmp6 >= tmp9;
                            auto tmp43 = static_cast<long>(832);
                            auto tmp44 = tmp6 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr3[static_cast<long>(256L + x2 + (1088L*x1) + (213248L*x0))];
                                return tmp46;
                            }
                            ;
                            auto tmp47 = tmp42 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp48 = tmp10 ? tmp41 : tmp47;
                            return tmp48;
                        }
                        ;
                        auto tmp49 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp50 = tmp0 >= tmp3;
                        auto tmp51 = static_cast<long>(896);
                        auto tmp52 = tmp0 < tmp51;
                        auto tmp53 = [&]
                        {
                            auto tmp54 = in_ptr4[static_cast<long>(192L + x2 + (1088L*x1) + (213248L*x0))];
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp50 ? tmp53() : static_cast<decltype(tmp53())>(0.0);
                        auto tmp56 = tmp4 ? tmp49 : tmp55;
                        out_ptr1[static_cast<long>(x1 + (196L*x2) + (376320L*x0))] = tmp56;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (376320L*x0)));
                            auto tmp1 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp13 = in_ptr9[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr10[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(0.001);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = std::sqrt(tmp6);
                            auto tmp8 = 1 / tmp7;
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp3 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                            tmp19.store(tmp20 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp20, 8, out_ptr2 + static_cast<long>(x1 + (1920L*x2) + (376320L*x0)), static_cast<long>(1920L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (376320L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(0.001);
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
                        tmp17.store(out_ptr2 + static_cast<long>(x1 + (1920L*x2) + (376320L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_74 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1984L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp31 = in_ptr3[static_cast<long>(x1)];
                        auto tmp33 = in_ptr4[static_cast<long>(x1)];
                        auto tmp41 = in_ptr5[static_cast<long>(x1)];
                        auto tmp43 = in_ptr6[static_cast<long>(x1)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (376320L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x1 + (1088L*x2) + (213248L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(1984);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = c10::convert<long>((-1024L) + x1);
                            auto tmp15 = static_cast<long>(0);
                            auto tmp16 = tmp14 >= tmp15;
                            auto tmp17 = static_cast<long>(896);
                            auto tmp18 = tmp14 < tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = in_ptr2[static_cast<long>((-200704L) + x2 + (196L*x1) + (376320L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                            auto tmp22 = tmp14 >= tmp17;
                            auto tmp23 = static_cast<long>(960);
                            auto tmp24 = tmp14 < tmp23;
                            auto tmp25 = [&]
                            {
                                auto tmp26 = in_ptr1[static_cast<long>((-896L) + x1 + (1088L*x2) + (213248L*x0))];
                                return tmp26;
                            }
                            ;
                            auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                            auto tmp28 = tmp18 ? tmp21 : tmp27;
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp10 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp30 = tmp4 ? tmp9 : tmp29;
                        auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                        auto tmp34 = static_cast<float>(0.001);
                        auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                        auto tmp36 = std::sqrt(tmp35);
                        auto tmp37 = 1 / tmp36;
                        auto tmp38 = static_cast<float>(1.0);
                        auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                        auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                        auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                        auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                        auto tmp45 = tmp44 * (tmp44>0);
                        out_ptr0[static_cast<long>(x1 + (1984L*x2) + (388864L*x0))] = tmp45;
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_79 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2048L); x2+=static_cast<long>(1L))
                    {
                        auto tmp47 = in_ptr4[static_cast<long>(x2)];
                        auto tmp49 = in_ptr5[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (376320L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<long>(2048);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = c10::convert<long>((-1024L) + x2);
                            auto tmp17 = static_cast<long>(0);
                            auto tmp18 = tmp16 >= tmp17;
                            auto tmp19 = static_cast<long>(960);
                            auto tmp20 = tmp16 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = c10::convert<long>((-1024L) + x2);
                                auto tmp23 = static_cast<long>(0);
                                auto tmp24 = tmp22 >= tmp23;
                                auto tmp25 = static_cast<long>(896);
                                auto tmp26 = tmp22 < tmp25;
                                auto tmp27 = [&]
                                {
                                    auto tmp28 = in_ptr3[static_cast<long>((-200704L) + x1 + (196L*x2) + (376320L*x0))];
                                    return tmp28;
                                }
                                ;
                                auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                                auto tmp30 = tmp22 >= tmp25;
                                auto tmp31 = static_cast<long>(960);
                                auto tmp32 = tmp22 < tmp31;
                                auto tmp33 = [&]
                                {
                                    auto tmp34 = in_ptr1[static_cast<long>((-896L) + x2 + (1088L*x1) + (213248L*x0))];
                                    return tmp34;
                                }
                                ;
                                auto tmp35 = tmp30 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                                auto tmp36 = tmp26 ? tmp29 : tmp35;
                                return tmp36;
                            }
                            ;
                            auto tmp37 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp38 = tmp16 >= tmp19;
                            auto tmp39 = static_cast<long>(1024);
                            auto tmp40 = tmp16 < tmp39;
                            auto tmp41 = [&]
                            {
                                auto tmp42 = in_ptr2[static_cast<long>((-960L) + x2 + (1088L*x1) + (213248L*x0))];
                                return tmp42;
                            }
                            ;
                            auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                            auto tmp44 = tmp20 ? tmp37 : tmp43;
                            return tmp44;
                        }
                        ;
                        auto tmp45 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp46 = tmp4 ? tmp11 : tmp45;
                        auto tmp48 = decltype(tmp46)(tmp46 - tmp47);
                        auto tmp50 = static_cast<float>(0.001);
                        auto tmp51 = decltype(tmp49)(tmp49 + tmp50);
                        auto tmp52 = std::sqrt(tmp51);
                        auto tmp53 = 1 / tmp52;
                        auto tmp54 = static_cast<float>(1.0);
                        auto tmp55 = decltype(tmp53)(tmp53 * tmp54);
                        auto tmp56 = decltype(tmp48)(tmp48 * tmp55);
                        out_ptr0[static_cast<long>(x2 + (2048L*x1) + (401408L*x0))] = tmp56;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = at::vec::clamp_min(tmp4, decltype(tmp4)(0));
                    tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_80 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_82 = async_compile.cpp('''
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
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2112L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (376320L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            auto tmp11 = in_ptr3[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp14 = tmp0 >= tmp3;
                        auto tmp15 = static_cast<long>(2112);
                        auto tmp16 = tmp0 < tmp15;
                        auto tmp17 = [&]
                        {
                            auto tmp18 = c10::convert<long>((-1024L) + x2);
                            auto tmp19 = static_cast<long>(0);
                            auto tmp20 = tmp18 >= tmp19;
                            auto tmp21 = static_cast<long>(1024);
                            auto tmp22 = tmp18 < tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>((-1024L) + x2);
                                auto tmp25 = static_cast<long>(0);
                                auto tmp26 = tmp24 >= tmp25;
                                auto tmp27 = static_cast<long>(960);
                                auto tmp28 = tmp24 < tmp27;
                                auto tmp29 = [&]
                                {
                                    auto tmp30 = c10::convert<long>((-1024L) + x2);
                                    auto tmp31 = static_cast<long>(0);
                                    auto tmp32 = tmp30 >= tmp31;
                                    auto tmp33 = static_cast<long>(896);
                                    auto tmp34 = tmp30 < tmp33;
                                    auto tmp35 = [&]
                                    {
                                        auto tmp36 = in_ptr4[static_cast<long>((-200704L) + x1 + (196L*x2) + (376320L*x0))];
                                        return tmp36;
                                    }
                                    ;
                                    auto tmp37 = tmp34 ? tmp35() : static_cast<decltype(tmp35())>(0.0);
                                    auto tmp38 = tmp30 >= tmp33;
                                    auto tmp39 = static_cast<long>(960);
                                    auto tmp40 = tmp30 < tmp39;
                                    auto tmp41 = [&]
                                    {
                                        auto tmp42 = in_ptr1[static_cast<long>((-896L) + x2 + (1088L*x1) + (213248L*x0))];
                                        return tmp42;
                                    }
                                    ;
                                    auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                                    auto tmp44 = tmp34 ? tmp37 : tmp43;
                                    return tmp44;
                                }
                                ;
                                auto tmp45 = tmp28 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                                auto tmp46 = tmp24 >= tmp27;
                                auto tmp47 = static_cast<long>(1024);
                                auto tmp48 = tmp24 < tmp47;
                                auto tmp49 = [&]
                                {
                                    auto tmp50 = in_ptr2[static_cast<long>((-960L) + x2 + (1088L*x1) + (213248L*x0))];
                                    return tmp50;
                                }
                                ;
                                auto tmp51 = tmp46 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                auto tmp52 = tmp28 ? tmp45 : tmp51;
                                return tmp52;
                            }
                            ;
                            auto tmp53 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp54 = tmp18 >= tmp21;
                            auto tmp55 = static_cast<long>(1088);
                            auto tmp56 = tmp18 < tmp55;
                            auto tmp57 = [&]
                            {
                                auto tmp58 = in_ptr3[static_cast<long>((-1024L) + x2 + (1088L*x1) + (213248L*x0))];
                                return tmp58;
                            }
                            ;
                            auto tmp59 = tmp54 ? tmp57() : static_cast<decltype(tmp57())>(0.0);
                            auto tmp60 = tmp22 ? tmp53 : tmp59;
                            return tmp60;
                        }
                        ;
                        auto tmp61 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                        auto tmp62 = tmp4 ? tmp13 : tmp61;
                        out_ptr0[static_cast<long>(x2 + (2112L*x1) + (413952L*x0))] = tmp62;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (2112L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2112L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_83 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_relu_85 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (376320L*x0)), static_cast<long>(196L), tmp0, 8);
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1088L*x1) + (1088L*x1_inner) + (213248L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1088L*x1) + (1088L*x1_inner) + (213248L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1088L*x1) + (1088L*x1_inner) + (213248L*x0)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1088L*x1) + (1088L*x1_inner) + (213248L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = tmp5 + tmp6;
                            auto tmp9 = tmp7 + tmp8;
                            tmp9.store(tmp10 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp10, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (426496L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (376320L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                        auto tmp7 = in_ptr4[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (426496L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1088);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = c10::convert<long>(x2);
                            auto tmp7 = static_cast<long>(0);
                            auto tmp8 = tmp6 >= tmp7;
                            auto tmp9 = static_cast<long>(1024);
                            auto tmp10 = tmp6 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(x2);
                                auto tmp13 = static_cast<long>(0);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = static_cast<long>(960);
                                auto tmp16 = tmp12 < tmp15;
                                auto tmp17 = [&]
                                {
                                    auto tmp18 = c10::convert<long>(x2);
                                    auto tmp19 = static_cast<long>(0);
                                    auto tmp20 = tmp18 >= tmp19;
                                    auto tmp21 = static_cast<long>(896);
                                    auto tmp22 = tmp18 < tmp21;
                                    auto tmp23 = [&]
                                    {
                                        auto tmp24 = in_ptr5[static_cast<long>(x1 + (196L*x2) + (376320L*x0))];
                                        return tmp24;
                                    }
                                    ;
                                    auto tmp25 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                                    auto tmp26 = tmp18 >= tmp21;
                                    auto tmp27 = static_cast<long>(960);
                                    auto tmp28 = tmp18 < tmp27;
                                    auto tmp29 = [&]
                                    {
                                        auto tmp30 = in_ptr1[static_cast<long>(128L + x2 + (1088L*x1) + (213248L*x0))];
                                        return tmp30;
                                    }
                                    ;
                                    auto tmp31 = tmp26 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                                    auto tmp32 = tmp22 ? tmp25 : tmp31;
                                    return tmp32;
                                }
                                ;
                                auto tmp33 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                                auto tmp34 = tmp12 >= tmp15;
                                auto tmp35 = static_cast<long>(1024);
                                auto tmp36 = tmp12 < tmp35;
                                auto tmp37 = [&]
                                {
                                    auto tmp38 = in_ptr2[static_cast<long>(64L + x2 + (1088L*x1) + (213248L*x0))];
                                    return tmp38;
                                }
                                ;
                                auto tmp39 = tmp34 ? tmp37() : static_cast<decltype(tmp37())>(0.0);
                                auto tmp40 = tmp16 ? tmp33 : tmp39;
                                return tmp40;
                            }
                            ;
                            auto tmp41 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp42 = tmp6 >= tmp9;
                            auto tmp43 = static_cast<long>(1088);
                            auto tmp44 = tmp6 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr3[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                                return tmp46;
                            }
                            ;
                            auto tmp47 = tmp42 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp48 = tmp10 ? tmp41 : tmp47;
                            return tmp48;
                        }
                        ;
                        auto tmp49 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp50 = tmp0 >= tmp3;
                        auto tmp51 = static_cast<long>(1152);
                        auto tmp52 = tmp0 < tmp51;
                        auto tmp53 = [&]
                        {
                            auto tmp54 = in_ptr4[static_cast<long>((-64L) + x2 + (1088L*x1) + (213248L*x0))];
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp50 ? tmp53() : static_cast<decltype(tmp53())>(0.0);
                        auto tmp56 = tmp4 ? tmp49 : tmp55;
                        out_ptr1[static_cast<long>(x1 + (196L*x2) + (426496L*x0))] = tmp56;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2176L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (426496L*x0)));
                            auto tmp1 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp13 = in_ptr9[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr10[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(0.001);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = std::sqrt(tmp6);
                            auto tmp8 = 1 / tmp7;
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp3 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                            tmp19.store(tmp20 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp20, 8, out_ptr2 + static_cast<long>(x1 + (2176L*x2) + (426496L*x0)), static_cast<long>(2176L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (426496L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(0.001);
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
                        tmp17.store(out_ptr2 + static_cast<long>(x1 + (2176L*x2) + (426496L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_86 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2240L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp31 = in_ptr3[static_cast<long>(x1)];
                        auto tmp33 = in_ptr4[static_cast<long>(x1)];
                        auto tmp41 = in_ptr5[static_cast<long>(x1)];
                        auto tmp43 = in_ptr6[static_cast<long>(x1)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (426496L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x1 + (1088L*x2) + (213248L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(2240);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = c10::convert<long>((-1024L) + x1);
                            auto tmp15 = static_cast<long>(0);
                            auto tmp16 = tmp14 >= tmp15;
                            auto tmp17 = static_cast<long>(1152);
                            auto tmp18 = tmp14 < tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = in_ptr2[static_cast<long>((-200704L) + x2 + (196L*x1) + (426496L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                            auto tmp22 = tmp14 >= tmp17;
                            auto tmp23 = static_cast<long>(1216);
                            auto tmp24 = tmp14 < tmp23;
                            auto tmp25 = [&]
                            {
                                auto tmp26 = in_ptr1[static_cast<long>((-1152L) + x1 + (1088L*x2) + (213248L*x0))];
                                return tmp26;
                            }
                            ;
                            auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                            auto tmp28 = tmp18 ? tmp21 : tmp27;
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp10 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp30 = tmp4 ? tmp9 : tmp29;
                        auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                        auto tmp34 = static_cast<float>(0.001);
                        auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                        auto tmp36 = std::sqrt(tmp35);
                        auto tmp37 = 1 / tmp36;
                        auto tmp38 = static_cast<float>(1.0);
                        auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                        auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                        auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                        auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                        auto tmp45 = tmp44 * (tmp44>0);
                        out_ptr0[static_cast<long>(x1 + (2240L*x2) + (439040L*x0))] = tmp45;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_89 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_91 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                    {
                        auto tmp47 = in_ptr4[static_cast<long>(x2)];
                        auto tmp49 = in_ptr5[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (426496L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<long>(2304);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = c10::convert<long>((-1024L) + x2);
                            auto tmp17 = static_cast<long>(0);
                            auto tmp18 = tmp16 >= tmp17;
                            auto tmp19 = static_cast<long>(1216);
                            auto tmp20 = tmp16 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = c10::convert<long>((-1024L) + x2);
                                auto tmp23 = static_cast<long>(0);
                                auto tmp24 = tmp22 >= tmp23;
                                auto tmp25 = static_cast<long>(1152);
                                auto tmp26 = tmp22 < tmp25;
                                auto tmp27 = [&]
                                {
                                    auto tmp28 = in_ptr3[static_cast<long>((-200704L) + x1 + (196L*x2) + (426496L*x0))];
                                    return tmp28;
                                }
                                ;
                                auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                                auto tmp30 = tmp22 >= tmp25;
                                auto tmp31 = static_cast<long>(1216);
                                auto tmp32 = tmp22 < tmp31;
                                auto tmp33 = [&]
                                {
                                    auto tmp34 = in_ptr1[static_cast<long>((-1152L) + x2 + (1088L*x1) + (213248L*x0))];
                                    return tmp34;
                                }
                                ;
                                auto tmp35 = tmp30 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                                auto tmp36 = tmp26 ? tmp29 : tmp35;
                                return tmp36;
                            }
                            ;
                            auto tmp37 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp38 = tmp16 >= tmp19;
                            auto tmp39 = static_cast<long>(1280);
                            auto tmp40 = tmp16 < tmp39;
                            auto tmp41 = [&]
                            {
                                auto tmp42 = in_ptr2[static_cast<long>((-1216L) + x2 + (1088L*x1) + (213248L*x0))];
                                return tmp42;
                            }
                            ;
                            auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                            auto tmp44 = tmp20 ? tmp37 : tmp43;
                            return tmp44;
                        }
                        ;
                        auto tmp45 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp46 = tmp4 ? tmp11 : tmp45;
                        auto tmp48 = decltype(tmp46)(tmp46 - tmp47);
                        auto tmp50 = static_cast<float>(0.001);
                        auto tmp51 = decltype(tmp49)(tmp49 + tmp50);
                        auto tmp52 = std::sqrt(tmp51);
                        auto tmp53 = 1 / tmp52;
                        auto tmp54 = static_cast<float>(1.0);
                        auto tmp55 = decltype(tmp53)(tmp53 * tmp54);
                        auto tmp56 = decltype(tmp48)(tmp48 * tmp55);
                        out_ptr0[static_cast<long>(x2 + (2304L*x1) + (451584L*x0))] = tmp56;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (2304L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = at::vec::clamp_min(tmp4, decltype(tmp4)(0));
                    tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (2304L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_92 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_94 = async_compile.cpp('''
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
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2368L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (426496L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            auto tmp11 = in_ptr3[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp14 = tmp0 >= tmp3;
                        auto tmp15 = static_cast<long>(2368);
                        auto tmp16 = tmp0 < tmp15;
                        auto tmp17 = [&]
                        {
                            auto tmp18 = c10::convert<long>((-1024L) + x2);
                            auto tmp19 = static_cast<long>(0);
                            auto tmp20 = tmp18 >= tmp19;
                            auto tmp21 = static_cast<long>(1280);
                            auto tmp22 = tmp18 < tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>((-1024L) + x2);
                                auto tmp25 = static_cast<long>(0);
                                auto tmp26 = tmp24 >= tmp25;
                                auto tmp27 = static_cast<long>(1216);
                                auto tmp28 = tmp24 < tmp27;
                                auto tmp29 = [&]
                                {
                                    auto tmp30 = c10::convert<long>((-1024L) + x2);
                                    auto tmp31 = static_cast<long>(0);
                                    auto tmp32 = tmp30 >= tmp31;
                                    auto tmp33 = static_cast<long>(1152);
                                    auto tmp34 = tmp30 < tmp33;
                                    auto tmp35 = [&]
                                    {
                                        auto tmp36 = in_ptr4[static_cast<long>((-200704L) + x1 + (196L*x2) + (426496L*x0))];
                                        return tmp36;
                                    }
                                    ;
                                    auto tmp37 = tmp34 ? tmp35() : static_cast<decltype(tmp35())>(0.0);
                                    auto tmp38 = tmp30 >= tmp33;
                                    auto tmp39 = static_cast<long>(1216);
                                    auto tmp40 = tmp30 < tmp39;
                                    auto tmp41 = [&]
                                    {
                                        auto tmp42 = in_ptr1[static_cast<long>((-1152L) + x2 + (1088L*x1) + (213248L*x0))];
                                        return tmp42;
                                    }
                                    ;
                                    auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                                    auto tmp44 = tmp34 ? tmp37 : tmp43;
                                    return tmp44;
                                }
                                ;
                                auto tmp45 = tmp28 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                                auto tmp46 = tmp24 >= tmp27;
                                auto tmp47 = static_cast<long>(1280);
                                auto tmp48 = tmp24 < tmp47;
                                auto tmp49 = [&]
                                {
                                    auto tmp50 = in_ptr2[static_cast<long>((-1216L) + x2 + (1088L*x1) + (213248L*x0))];
                                    return tmp50;
                                }
                                ;
                                auto tmp51 = tmp46 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                auto tmp52 = tmp28 ? tmp45 : tmp51;
                                return tmp52;
                            }
                            ;
                            auto tmp53 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp54 = tmp18 >= tmp21;
                            auto tmp55 = static_cast<long>(1344);
                            auto tmp56 = tmp18 < tmp55;
                            auto tmp57 = [&]
                            {
                                auto tmp58 = in_ptr3[static_cast<long>((-1280L) + x2 + (1088L*x1) + (213248L*x0))];
                                return tmp58;
                            }
                            ;
                            auto tmp59 = tmp54 ? tmp57() : static_cast<decltype(tmp57())>(0.0);
                            auto tmp60 = tmp22 ? tmp53 : tmp59;
                            return tmp60;
                        }
                        ;
                        auto tmp61 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                        auto tmp62 = tmp4 ? tmp13 : tmp61;
                        out_ptr0[static_cast<long>(x2 + (2368L*x1) + (464128L*x0))] = tmp62;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (2368L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_95 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_97 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1408L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1344);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = c10::convert<long>(x2);
                            auto tmp7 = static_cast<long>(0);
                            auto tmp8 = tmp6 >= tmp7;
                            auto tmp9 = static_cast<long>(1280);
                            auto tmp10 = tmp6 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(x2);
                                auto tmp13 = static_cast<long>(0);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = static_cast<long>(1216);
                                auto tmp16 = tmp12 < tmp15;
                                auto tmp17 = [&]
                                {
                                    auto tmp18 = c10::convert<long>(x2);
                                    auto tmp19 = static_cast<long>(0);
                                    auto tmp20 = tmp18 >= tmp19;
                                    auto tmp21 = static_cast<long>(1152);
                                    auto tmp22 = tmp18 < tmp21;
                                    auto tmp23 = [&]
                                    {
                                        auto tmp24 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (426496L*x0))];
                                        return tmp24;
                                    }
                                    ;
                                    auto tmp25 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                                    auto tmp26 = tmp18 >= tmp21;
                                    auto tmp27 = static_cast<long>(1216);
                                    auto tmp28 = tmp18 < tmp27;
                                    auto tmp29 = [&]
                                    {
                                        auto tmp30 = in_ptr1[static_cast<long>((-128L) + x2 + (1088L*x1) + (213248L*x0))];
                                        return tmp30;
                                    }
                                    ;
                                    auto tmp31 = tmp26 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                                    auto tmp32 = tmp22 ? tmp25 : tmp31;
                                    return tmp32;
                                }
                                ;
                                auto tmp33 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                                auto tmp34 = tmp12 >= tmp15;
                                auto tmp35 = static_cast<long>(1280);
                                auto tmp36 = tmp12 < tmp35;
                                auto tmp37 = [&]
                                {
                                    auto tmp38 = in_ptr2[static_cast<long>((-192L) + x2 + (1088L*x1) + (213248L*x0))];
                                    return tmp38;
                                }
                                ;
                                auto tmp39 = tmp34 ? tmp37() : static_cast<decltype(tmp37())>(0.0);
                                auto tmp40 = tmp16 ? tmp33 : tmp39;
                                return tmp40;
                            }
                            ;
                            auto tmp41 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp42 = tmp6 >= tmp9;
                            auto tmp43 = static_cast<long>(1344);
                            auto tmp44 = tmp6 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr3[static_cast<long>((-256L) + x2 + (1088L*x1) + (213248L*x0))];
                                return tmp46;
                            }
                            ;
                            auto tmp47 = tmp42 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp48 = tmp10 ? tmp41 : tmp47;
                            return tmp48;
                        }
                        ;
                        auto tmp49 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp50 = tmp0 >= tmp3;
                        auto tmp51 = static_cast<long>(1408);
                        auto tmp52 = tmp0 < tmp51;
                        auto tmp53 = [&]
                        {
                            auto tmp54 = in_ptr4[static_cast<long>((-320L) + x2 + (1088L*x1) + (213248L*x0))];
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp50 ? tmp53() : static_cast<decltype(tmp53())>(0.0);
                        auto tmp56 = tmp4 ? tmp49 : tmp55;
                        out_ptr0[static_cast<long>(x2 + (1408L*x1) + (275968L*x0))] = tmp56;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2432L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr5[static_cast<long>(x1 + (196L*x2) + (426496L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            auto tmp11 = in_ptr3[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                            auto tmp13 = in_ptr4[static_cast<long>(x2 + (1088L*x1) + (213248L*x0))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp16 = tmp0 >= tmp3;
                        auto tmp17 = static_cast<long>(2432);
                        auto tmp18 = tmp0 < tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = out_ptr0[static_cast<long>((-1024L) + x2 + (1408L*x1) + (275968L*x0))];
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp16 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp22 = tmp4 ? tmp15 : tmp21;
                        out_ptr1[static_cast<long>(x2 + (2432L*x1) + (476672L*x0))] = tmp22;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2432L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (2432L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    auto tmp19 = tmp0 - tmp18;
                    auto tmp21 = tmp20 + tmp5;
                    auto tmp22 = tmp21.sqrt();
                    auto tmp23 = tmp22.reciprocal();
                    auto tmp24 = tmp23 * tmp10;
                    auto tmp25 = tmp19 * tmp24;
                    auto tmp27 = tmp25 * tmp26;
                    auto tmp29 = tmp27 + tmp28;
                    auto tmp30 = at::vec::clamp_min(tmp29, decltype(tmp29)(0));
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2432L*x0)));
                    tmp30.store(out_ptr3 + static_cast<long>(x1 + (2432L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_98 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1600L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1600L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1600L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2432L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr2[static_cast<long>(x1)];
                    auto tmp33 = in_ptr3[static_cast<long>(x1)];
                    auto tmp41 = in_ptr4[static_cast<long>(x1)];
                    auto tmp43 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(2048);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (2304L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (2176L*x0))];
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        return tmp8;
                    }
                    ;
                    auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp10 = tmp0 >= tmp3;
                    auto tmp11 = static_cast<long>(2432);
                    auto tmp12 = tmp0 < tmp11;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = c10::convert<long>((-2048L) + x1);
                        auto tmp15 = static_cast<long>(0);
                        auto tmp16 = tmp14 >= tmp15;
                        auto tmp17 = static_cast<long>(256);
                        auto tmp18 = tmp14 < tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = in_ptr0[static_cast<long>(x1 + (2304L*x0))];
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp22 = tmp14 >= tmp17;
                        auto tmp23 = static_cast<long>(384);
                        auto tmp24 = tmp14 < tmp23;
                        auto tmp25 = [&]
                        {
                            auto tmp26 = in_ptr1[static_cast<long>((-256L) + x1 + (2176L*x0))];
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                        auto tmp28 = tmp18 ? tmp21 : tmp27;
                        return tmp28;
                    }
                    ;
                    auto tmp29 = tmp10 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp30 = tmp4 ? tmp9 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(0.001);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    auto tmp45 = tmp44 * (tmp44>0);
                    out_ptr0[static_cast<long>(x1 + (2432L*x0))] = tmp45;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_101 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1600L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1600L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1600L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(1L))
                {
                    auto tmp47 = in_ptr3[static_cast<long>(x1)];
                    auto tmp49 = in_ptr4[static_cast<long>(x1)];
                    auto tmp57 = in_ptr5[static_cast<long>(x1)];
                    auto tmp59 = in_ptr6[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(2048);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (2304L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (2176L*x0))];
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = in_ptr2[static_cast<long>(x1 + (2176L*x0))];
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(2560);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = c10::convert<long>((-2048L) + x1);
                        auto tmp17 = static_cast<long>(0);
                        auto tmp18 = tmp16 >= tmp17;
                        auto tmp19 = static_cast<long>(384);
                        auto tmp20 = tmp16 < tmp19;
                        auto tmp21 = [&]
                        {
                            auto tmp22 = c10::convert<long>((-2048L) + x1);
                            auto tmp23 = static_cast<long>(0);
                            auto tmp24 = tmp22 >= tmp23;
                            auto tmp25 = static_cast<long>(256);
                            auto tmp26 = tmp22 < tmp25;
                            auto tmp27 = [&]
                            {
                                auto tmp28 = in_ptr0[static_cast<long>(x1 + (2304L*x0))];
                                return tmp28;
                            }
                            ;
                            auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                            auto tmp30 = tmp22 >= tmp25;
                            auto tmp31 = static_cast<long>(384);
                            auto tmp32 = tmp22 < tmp31;
                            auto tmp33 = [&]
                            {
                                auto tmp34 = in_ptr1[static_cast<long>((-256L) + x1 + (2176L*x0))];
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp30 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                            auto tmp36 = tmp26 ? tmp29 : tmp35;
                            return tmp36;
                        }
                        ;
                        auto tmp37 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                        auto tmp38 = tmp16 >= tmp19;
                        auto tmp39 = static_cast<long>(512);
                        auto tmp40 = tmp16 < tmp39;
                        auto tmp41 = [&]
                        {
                            auto tmp42 = in_ptr2[static_cast<long>((-384L) + x1 + (2176L*x0))];
                            return tmp42;
                        }
                        ;
                        auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                        auto tmp44 = tmp20 ? tmp37 : tmp43;
                        return tmp44;
                    }
                    ;
                    auto tmp45 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp46 = tmp4 ? tmp11 : tmp45;
                    auto tmp48 = decltype(tmp46)(tmp46 - tmp47);
                    auto tmp50 = static_cast<float>(0.001);
                    auto tmp51 = decltype(tmp49)(tmp49 + tmp50);
                    auto tmp52 = std::sqrt(tmp51);
                    auto tmp53 = 1 / tmp52;
                    auto tmp54 = static_cast<float>(1.0);
                    auto tmp55 = decltype(tmp53)(tmp53 * tmp54);
                    auto tmp56 = decltype(tmp48)(tmp48 * tmp55);
                    auto tmp58 = decltype(tmp56)(tmp56 * tmp57);
                    auto tmp60 = decltype(tmp58)(tmp58 + tmp59);
                    auto tmp61 = tmp60 * (tmp60>0);
                    in_out_ptr0[static_cast<long>(x1 + (2560L*x0))] = tmp61;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_104 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1600L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1600L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1600L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_relu_106 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2688L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(2048);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (2304L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (2176L*x0))];
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = in_ptr2[static_cast<long>(x1 + (2176L*x0))];
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = in_ptr3[static_cast<long>(x1 + (2176L*x0))];
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp14 = tmp0 >= tmp3;
                    auto tmp15 = static_cast<long>(2688);
                    auto tmp16 = tmp0 < tmp15;
                    auto tmp17 = [&]
                    {
                        auto tmp18 = c10::convert<long>((-2048L) + x1);
                        auto tmp19 = static_cast<long>(0);
                        auto tmp20 = tmp18 >= tmp19;
                        auto tmp21 = static_cast<long>(512);
                        auto tmp22 = tmp18 < tmp21;
                        auto tmp23 = [&]
                        {
                            auto tmp24 = c10::convert<long>((-2048L) + x1);
                            auto tmp25 = static_cast<long>(0);
                            auto tmp26 = tmp24 >= tmp25;
                            auto tmp27 = static_cast<long>(384);
                            auto tmp28 = tmp24 < tmp27;
                            auto tmp29 = [&]
                            {
                                auto tmp30 = c10::convert<long>((-2048L) + x1);
                                auto tmp31 = static_cast<long>(0);
                                auto tmp32 = tmp30 >= tmp31;
                                auto tmp33 = static_cast<long>(256);
                                auto tmp34 = tmp30 < tmp33;
                                auto tmp35 = [&]
                                {
                                    auto tmp36 = in_ptr0[static_cast<long>(x1 + (2304L*x0))];
                                    return tmp36;
                                }
                                ;
                                auto tmp37 = tmp34 ? tmp35() : static_cast<decltype(tmp35())>(0.0);
                                auto tmp38 = tmp30 >= tmp33;
                                auto tmp39 = static_cast<long>(384);
                                auto tmp40 = tmp30 < tmp39;
                                auto tmp41 = [&]
                                {
                                    auto tmp42 = in_ptr1[static_cast<long>((-256L) + x1 + (2176L*x0))];
                                    return tmp42;
                                }
                                ;
                                auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                                auto tmp44 = tmp34 ? tmp37 : tmp43;
                                return tmp44;
                            }
                            ;
                            auto tmp45 = tmp28 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                            auto tmp46 = tmp24 >= tmp27;
                            auto tmp47 = static_cast<long>(512);
                            auto tmp48 = tmp24 < tmp47;
                            auto tmp49 = [&]
                            {
                                auto tmp50 = in_ptr2[static_cast<long>((-384L) + x1 + (2176L*x0))];
                                return tmp50;
                            }
                            ;
                            auto tmp51 = tmp46 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                            auto tmp52 = tmp28 ? tmp45 : tmp51;
                            return tmp52;
                        }
                        ;
                        auto tmp53 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                        auto tmp54 = tmp18 >= tmp21;
                        auto tmp55 = static_cast<long>(640);
                        auto tmp56 = tmp18 < tmp55;
                        auto tmp57 = [&]
                        {
                            auto tmp58 = in_ptr3[static_cast<long>((-512L) + x1 + (2176L*x0))];
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp54 ? tmp57() : static_cast<decltype(tmp57())>(0.0);
                        auto tmp60 = tmp22 ? tmp53 : tmp59;
                        return tmp60;
                    }
                    ;
                    auto tmp61 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                    auto tmp62 = tmp4 ? tmp13 : tmp61;
                    out_ptr0[static_cast<long>(x1 + (2688L*x0))] = tmp62;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2688L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (2688L*x2) + (131712L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(0.001);
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
                            tmp_acc0_vec = tmp_acc0_vec + tmp17;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (2688L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(21504L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, ), (1, ))
    assert_size_stride(arg1_1, (128, ), (1, ))
    assert_size_stride(arg2_1, (128, ), (1, ))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (200, ), (1, ))
    assert_size_stride(arg7_1, (200, ), (1, ))
    assert_size_stride(arg8_1, (200, ), (1, ))
    assert_size_stride(arg9_1, (200, ), (1, ))
    assert_size_stride(arg10_1, (316, ), (1, ))
    assert_size_stride(arg11_1, (316, ), (1, ))
    assert_size_stride(arg12_1, (200, ), (1, ))
    assert_size_stride(arg13_1, (200, ), (1, ))
    assert_size_stride(arg14_1, (200, ), (1, ))
    assert_size_stride(arg15_1, (200, ), (1, ))
    assert_size_stride(arg16_1, (336, ), (1, ))
    assert_size_stride(arg17_1, (336, ), (1, ))
    assert_size_stride(arg18_1, (200, ), (1, ))
    assert_size_stride(arg19_1, (200, ), (1, ))
    assert_size_stride(arg20_1, (200, ), (1, ))
    assert_size_stride(arg21_1, (200, ), (1, ))
    assert_size_stride(arg22_1, (356, ), (1, ))
    assert_size_stride(arg23_1, (356, ), (1, ))
    assert_size_stride(arg24_1, (200, ), (1, ))
    assert_size_stride(arg25_1, (200, ), (1, ))
    assert_size_stride(arg26_1, (200, ), (1, ))
    assert_size_stride(arg27_1, (200, ), (1, ))
    assert_size_stride(arg28_1, (376, ), (1, ))
    assert_size_stride(arg29_1, (376, ), (1, ))
    assert_size_stride(arg30_1, (376, ), (1, ))
    assert_size_stride(arg31_1, (376, ), (1, ))
    assert_size_stride(arg32_1, (400, ), (1, ))
    assert_size_stride(arg33_1, (400, ), (1, ))
    assert_size_stride(arg34_1, (400, ), (1, ))
    assert_size_stride(arg35_1, (400, ), (1, ))
    assert_size_stride(arg36_1, (704, ), (1, ))
    assert_size_stride(arg37_1, (704, ), (1, ))
    assert_size_stride(arg38_1, (400, ), (1, ))
    assert_size_stride(arg39_1, (400, ), (1, ))
    assert_size_stride(arg40_1, (400, ), (1, ))
    assert_size_stride(arg41_1, (400, ), (1, ))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (400, ), (1, ))
    assert_size_stride(arg45_1, (400, ), (1, ))
    assert_size_stride(arg46_1, (400, ), (1, ))
    assert_size_stride(arg47_1, (400, ), (1, ))
    assert_size_stride(arg48_1, (832, ), (1, ))
    assert_size_stride(arg49_1, (832, ), (1, ))
    assert_size_stride(arg50_1, (400, ), (1, ))
    assert_size_stride(arg51_1, (400, ), (1, ))
    assert_size_stride(arg52_1, (400, ), (1, ))
    assert_size_stride(arg53_1, (400, ), (1, ))
    assert_size_stride(arg54_1, (896, ), (1, ))
    assert_size_stride(arg55_1, (896, ), (1, ))
    assert_size_stride(arg56_1, (400, ), (1, ))
    assert_size_stride(arg57_1, (400, ), (1, ))
    assert_size_stride(arg58_1, (400, ), (1, ))
    assert_size_stride(arg59_1, (400, ), (1, ))
    assert_size_stride(arg60_1, (960, ), (1, ))
    assert_size_stride(arg61_1, (960, ), (1, ))
    assert_size_stride(arg62_1, (400, ), (1, ))
    assert_size_stride(arg63_1, (400, ), (1, ))
    assert_size_stride(arg64_1, (400, ), (1, ))
    assert_size_stride(arg65_1, (400, ), (1, ))
    assert_size_stride(arg66_1, (1024, ), (1, ))
    assert_size_stride(arg67_1, (1024, ), (1, ))
    assert_size_stride(arg68_1, (400, ), (1, ))
    assert_size_stride(arg69_1, (400, ), (1, ))
    assert_size_stride(arg70_1, (400, ), (1, ))
    assert_size_stride(arg71_1, (400, ), (1, ))
    assert_size_stride(arg72_1, (1088, ), (1, ))
    assert_size_stride(arg73_1, (1088, ), (1, ))
    assert_size_stride(arg74_1, (400, ), (1, ))
    assert_size_stride(arg75_1, (400, ), (1, ))
    assert_size_stride(arg76_1, (400, ), (1, ))
    assert_size_stride(arg77_1, (400, ), (1, ))
    assert_size_stride(arg78_1, (1152, ), (1, ))
    assert_size_stride(arg79_1, (1152, ), (1, ))
    assert_size_stride(arg80_1, (1152, ), (1, ))
    assert_size_stride(arg81_1, (1152, ), (1, ))
    assert_size_stride(arg82_1, (800, ), (1, ))
    assert_size_stride(arg83_1, (800, ), (1, ))
    assert_size_stride(arg84_1, (800, ), (1, ))
    assert_size_stride(arg85_1, (800, ), (1, ))
    assert_size_stride(arg86_1, (1216, ), (1, ))
    assert_size_stride(arg87_1, (1216, ), (1, ))
    assert_size_stride(arg88_1, (800, ), (1, ))
    assert_size_stride(arg89_1, (800, ), (1, ))
    assert_size_stride(arg90_1, (800, ), (1, ))
    assert_size_stride(arg91_1, (800, ), (1, ))
    assert_size_stride(arg92_1, (1280, ), (1, ))
    assert_size_stride(arg93_1, (1280, ), (1, ))
    assert_size_stride(arg94_1, (800, ), (1, ))
    assert_size_stride(arg95_1, (800, ), (1, ))
    assert_size_stride(arg96_1, (800, ), (1, ))
    assert_size_stride(arg97_1, (800, ), (1, ))
    assert_size_stride(arg98_1, (1344, ), (1, ))
    assert_size_stride(arg99_1, (1344, ), (1, ))
    assert_size_stride(arg100_1, (800, ), (1, ))
    assert_size_stride(arg101_1, (800, ), (1, ))
    assert_size_stride(arg102_1, (800, ), (1, ))
    assert_size_stride(arg103_1, (800, ), (1, ))
    assert_size_stride(arg104_1, (1408, ), (1, ))
    assert_size_stride(arg105_1, (1408, ), (1, ))
    assert_size_stride(arg106_1, (800, ), (1, ))
    assert_size_stride(arg107_1, (800, ), (1, ))
    assert_size_stride(arg108_1, (800, ), (1, ))
    assert_size_stride(arg109_1, (800, ), (1, ))
    assert_size_stride(arg110_1, (1472, ), (1, ))
    assert_size_stride(arg111_1, (1472, ), (1, ))
    assert_size_stride(arg112_1, (800, ), (1, ))
    assert_size_stride(arg113_1, (800, ), (1, ))
    assert_size_stride(arg114_1, (800, ), (1, ))
    assert_size_stride(arg115_1, (800, ), (1, ))
    assert_size_stride(arg116_1, (1536, ), (1, ))
    assert_size_stride(arg117_1, (1536, ), (1, ))
    assert_size_stride(arg118_1, (800, ), (1, ))
    assert_size_stride(arg119_1, (800, ), (1, ))
    assert_size_stride(arg120_1, (800, ), (1, ))
    assert_size_stride(arg121_1, (800, ), (1, ))
    assert_size_stride(arg122_1, (1600, ), (1, ))
    assert_size_stride(arg123_1, (1600, ), (1, ))
    assert_size_stride(arg124_1, (800, ), (1, ))
    assert_size_stride(arg125_1, (800, ), (1, ))
    assert_size_stride(arg126_1, (800, ), (1, ))
    assert_size_stride(arg127_1, (800, ), (1, ))
    assert_size_stride(arg128_1, (1664, ), (1, ))
    assert_size_stride(arg129_1, (1664, ), (1, ))
    assert_size_stride(arg130_1, (800, ), (1, ))
    assert_size_stride(arg131_1, (800, ), (1, ))
    assert_size_stride(arg132_1, (800, ), (1, ))
    assert_size_stride(arg133_1, (800, ), (1, ))
    assert_size_stride(arg134_1, (1728, ), (1, ))
    assert_size_stride(arg135_1, (1728, ), (1, ))
    assert_size_stride(arg136_1, (800, ), (1, ))
    assert_size_stride(arg137_1, (800, ), (1, ))
    assert_size_stride(arg138_1, (800, ), (1, ))
    assert_size_stride(arg139_1, (800, ), (1, ))
    assert_size_stride(arg140_1, (1792, ), (1, ))
    assert_size_stride(arg141_1, (1792, ), (1, ))
    assert_size_stride(arg142_1, (800, ), (1, ))
    assert_size_stride(arg143_1, (800, ), (1, ))
    assert_size_stride(arg144_1, (800, ), (1, ))
    assert_size_stride(arg145_1, (800, ), (1, ))
    assert_size_stride(arg146_1, (1856, ), (1, ))
    assert_size_stride(arg147_1, (1856, ), (1, ))
    assert_size_stride(arg148_1, (800, ), (1, ))
    assert_size_stride(arg149_1, (800, ), (1, ))
    assert_size_stride(arg150_1, (800, ), (1, ))
    assert_size_stride(arg151_1, (800, ), (1, ))
    assert_size_stride(arg152_1, (1920, ), (1, ))
    assert_size_stride(arg153_1, (1920, ), (1, ))
    assert_size_stride(arg154_1, (800, ), (1, ))
    assert_size_stride(arg155_1, (800, ), (1, ))
    assert_size_stride(arg156_1, (800, ), (1, ))
    assert_size_stride(arg157_1, (800, ), (1, ))
    assert_size_stride(arg158_1, (1984, ), (1, ))
    assert_size_stride(arg159_1, (1984, ), (1, ))
    assert_size_stride(arg160_1, (800, ), (1, ))
    assert_size_stride(arg161_1, (800, ), (1, ))
    assert_size_stride(arg162_1, (800, ), (1, ))
    assert_size_stride(arg163_1, (800, ), (1, ))
    assert_size_stride(arg164_1, (2048, ), (1, ))
    assert_size_stride(arg165_1, (2048, ), (1, ))
    assert_size_stride(arg166_1, (800, ), (1, ))
    assert_size_stride(arg167_1, (800, ), (1, ))
    assert_size_stride(arg168_1, (800, ), (1, ))
    assert_size_stride(arg169_1, (800, ), (1, ))
    assert_size_stride(arg170_1, (2112, ), (1, ))
    assert_size_stride(arg171_1, (2112, ), (1, ))
    assert_size_stride(arg172_1, (800, ), (1, ))
    assert_size_stride(arg173_1, (800, ), (1, ))
    assert_size_stride(arg174_1, (800, ), (1, ))
    assert_size_stride(arg175_1, (800, ), (1, ))
    assert_size_stride(arg176_1, (2176, ), (1, ))
    assert_size_stride(arg177_1, (2176, ), (1, ))
    assert_size_stride(arg178_1, (800, ), (1, ))
    assert_size_stride(arg179_1, (800, ), (1, ))
    assert_size_stride(arg180_1, (800, ), (1, ))
    assert_size_stride(arg181_1, (800, ), (1, ))
    assert_size_stride(arg182_1, (2240, ), (1, ))
    assert_size_stride(arg183_1, (2240, ), (1, ))
    assert_size_stride(arg184_1, (800, ), (1, ))
    assert_size_stride(arg185_1, (800, ), (1, ))
    assert_size_stride(arg186_1, (800, ), (1, ))
    assert_size_stride(arg187_1, (800, ), (1, ))
    assert_size_stride(arg188_1, (2304, ), (1, ))
    assert_size_stride(arg189_1, (2304, ), (1, ))
    assert_size_stride(arg190_1, (800, ), (1, ))
    assert_size_stride(arg191_1, (800, ), (1, ))
    assert_size_stride(arg192_1, (800, ), (1, ))
    assert_size_stride(arg193_1, (800, ), (1, ))
    assert_size_stride(arg194_1, (2368, ), (1, ))
    assert_size_stride(arg195_1, (2368, ), (1, ))
    assert_size_stride(arg196_1, (800, ), (1, ))
    assert_size_stride(arg197_1, (800, ), (1, ))
    assert_size_stride(arg198_1, (800, ), (1, ))
    assert_size_stride(arg199_1, (800, ), (1, ))
    assert_size_stride(arg200_1, (2432, ), (1, ))
    assert_size_stride(arg201_1, (2432, ), (1, ))
    assert_size_stride(arg202_1, (2432, ), (1, ))
    assert_size_stride(arg203_1, (2432, ), (1, ))
    assert_size_stride(arg204_1, (1600, ), (1, ))
    assert_size_stride(arg205_1, (1600, ), (1, ))
    assert_size_stride(arg206_1, (1600, ), (1, ))
    assert_size_stride(arg207_1, (1600, ), (1, ))
    assert_size_stride(arg208_1, (2432, ), (1, ))
    assert_size_stride(arg209_1, (2432, ), (1, ))
    assert_size_stride(arg210_1, (1600, ), (1, ))
    assert_size_stride(arg211_1, (1600, ), (1, ))
    assert_size_stride(arg212_1, (1600, ), (1, ))
    assert_size_stride(arg213_1, (1600, ), (1, ))
    assert_size_stride(arg214_1, (2560, ), (1, ))
    assert_size_stride(arg215_1, (2560, ), (1, ))
    assert_size_stride(arg216_1, (1600, ), (1, ))
    assert_size_stride(arg217_1, (1600, ), (1, ))
    assert_size_stride(arg218_1, (1600, ), (1, ))
    assert_size_stride(arg219_1, (1600, ), (1, ))
    assert_size_stride(arg220_1, (2688, ), (1, ))
    assert_size_stride(arg221_1, (2688, ), (1, ))
    assert_size_stride(arg222_1, (128, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg223_1, (296, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg224_1, (200, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg225_1, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg226_1, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg227_1, (200, 316, 1, 1), (316, 1, 1, 1))
    assert_size_stride(arg228_1, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg229_1, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg230_1, (200, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg231_1, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg232_1, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg233_1, (200, 356, 1, 1), (356, 1, 1, 1))
    assert_size_stride(arg234_1, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg235_1, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg236_1, (640, 376, 1, 1), (376, 1, 1, 1))
    assert_size_stride(arg237_1, (400, 376, 1, 1), (376, 1, 1, 1))
    assert_size_stride(arg238_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg239_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg240_1, (400, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(arg241_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg242_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg243_1, (400, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg244_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg245_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg246_1, (400, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(arg247_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg248_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg249_1, (400, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg250_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg251_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg252_1, (400, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg253_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg254_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg255_1, (400, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg256_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg257_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg258_1, (400, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(arg259_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg260_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg261_1, (1152, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg262_1, (800, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg263_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg264_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg265_1, (800, 1216, 1, 1), (1216, 1, 1, 1))
    assert_size_stride(arg266_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg267_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg268_1, (800, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg269_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg270_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg271_1, (800, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(arg272_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg273_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg274_1, (800, 1408, 1, 1), (1408, 1, 1, 1))
    assert_size_stride(arg275_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg276_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg277_1, (800, 1472, 1, 1), (1472, 1, 1, 1))
    assert_size_stride(arg278_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg279_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg280_1, (800, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg281_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg282_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg283_1, (800, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(arg284_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg285_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg286_1, (800, 1664, 1, 1), (1664, 1, 1, 1))
    assert_size_stride(arg287_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg288_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg289_1, (800, 1728, 1, 1), (1728, 1, 1, 1))
    assert_size_stride(arg290_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg291_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg292_1, (800, 1792, 1, 1), (1792, 1, 1, 1))
    assert_size_stride(arg293_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg294_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg295_1, (800, 1856, 1, 1), (1856, 1, 1, 1))
    assert_size_stride(arg296_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg297_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg298_1, (800, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg299_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg300_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg301_1, (800, 1984, 1, 1), (1984, 1, 1, 1))
    assert_size_stride(arg302_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg303_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg304_1, (800, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg305_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg306_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg307_1, (800, 2112, 1, 1), (2112, 1, 1, 1))
    assert_size_stride(arg308_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg309_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg310_1, (800, 2176, 1, 1), (2176, 1, 1, 1))
    assert_size_stride(arg311_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg312_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg313_1, (800, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(arg314_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg315_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg316_1, (800, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(arg317_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg318_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg319_1, (800, 2368, 1, 1), (2368, 1, 1, 1))
    assert_size_stride(arg320_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg321_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg322_1, (2304, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(arg323_1, (1600, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(arg324_1, (1600, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg325_1, (2176, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(arg326_1, (1600, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(arg327_1, (1600, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg328_1, (2176, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(arg329_1, (1600, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(arg330_1, (1600, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg331_1, (2176, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(arg332_1, (1000, 2688, 1, 1), (2688, 1, 1, 1))
    assert_size_stride(arg333_1, (1000, ), (1, ))
    assert_size_stride(arg334_1, (128, ), (1, ))
    assert_size_stride(arg335_1, (128, ), (1, ))
    assert_size_stride(arg336_1, (128, ), (1, ))
    assert_size_stride(arg337_1, (128, ), (1, ))
    assert_size_stride(arg338_1, (128, ), (1, ))
    assert_size_stride(arg339_1, (128, ), (1, ))
    assert_size_stride(arg340_1, (200, ), (1, ))
    assert_size_stride(arg341_1, (200, ), (1, ))
    assert_size_stride(arg342_1, (200, ), (1, ))
    assert_size_stride(arg343_1, (200, ), (1, ))
    assert_size_stride(arg344_1, (316, ), (1, ))
    assert_size_stride(arg345_1, (316, ), (1, ))
    assert_size_stride(arg346_1, (200, ), (1, ))
    assert_size_stride(arg347_1, (200, ), (1, ))
    assert_size_stride(arg348_1, (200, ), (1, ))
    assert_size_stride(arg349_1, (200, ), (1, ))
    assert_size_stride(arg350_1, (336, ), (1, ))
    assert_size_stride(arg351_1, (336, ), (1, ))
    assert_size_stride(arg352_1, (200, ), (1, ))
    assert_size_stride(arg353_1, (200, ), (1, ))
    assert_size_stride(arg354_1, (200, ), (1, ))
    assert_size_stride(arg355_1, (200, ), (1, ))
    assert_size_stride(arg356_1, (356, ), (1, ))
    assert_size_stride(arg357_1, (356, ), (1, ))
    assert_size_stride(arg358_1, (200, ), (1, ))
    assert_size_stride(arg359_1, (200, ), (1, ))
    assert_size_stride(arg360_1, (200, ), (1, ))
    assert_size_stride(arg361_1, (200, ), (1, ))
    assert_size_stride(arg362_1, (376, ), (1, ))
    assert_size_stride(arg363_1, (376, ), (1, ))
    assert_size_stride(arg364_1, (376, ), (1, ))
    assert_size_stride(arg365_1, (376, ), (1, ))
    assert_size_stride(arg366_1, (400, ), (1, ))
    assert_size_stride(arg367_1, (400, ), (1, ))
    assert_size_stride(arg368_1, (400, ), (1, ))
    assert_size_stride(arg369_1, (400, ), (1, ))
    assert_size_stride(arg370_1, (704, ), (1, ))
    assert_size_stride(arg371_1, (704, ), (1, ))
    assert_size_stride(arg372_1, (400, ), (1, ))
    assert_size_stride(arg373_1, (400, ), (1, ))
    assert_size_stride(arg374_1, (400, ), (1, ))
    assert_size_stride(arg375_1, (400, ), (1, ))
    assert_size_stride(arg376_1, (768, ), (1, ))
    assert_size_stride(arg377_1, (768, ), (1, ))
    assert_size_stride(arg378_1, (400, ), (1, ))
    assert_size_stride(arg379_1, (400, ), (1, ))
    assert_size_stride(arg380_1, (400, ), (1, ))
    assert_size_stride(arg381_1, (400, ), (1, ))
    assert_size_stride(arg382_1, (832, ), (1, ))
    assert_size_stride(arg383_1, (832, ), (1, ))
    assert_size_stride(arg384_1, (400, ), (1, ))
    assert_size_stride(arg385_1, (400, ), (1, ))
    assert_size_stride(arg386_1, (400, ), (1, ))
    assert_size_stride(arg387_1, (400, ), (1, ))
    assert_size_stride(arg388_1, (896, ), (1, ))
    assert_size_stride(arg389_1, (896, ), (1, ))
    assert_size_stride(arg390_1, (400, ), (1, ))
    assert_size_stride(arg391_1, (400, ), (1, ))
    assert_size_stride(arg392_1, (400, ), (1, ))
    assert_size_stride(arg393_1, (400, ), (1, ))
    assert_size_stride(arg394_1, (960, ), (1, ))
    assert_size_stride(arg395_1, (960, ), (1, ))
    assert_size_stride(arg396_1, (400, ), (1, ))
    assert_size_stride(arg397_1, (400, ), (1, ))
    assert_size_stride(arg398_1, (400, ), (1, ))
    assert_size_stride(arg399_1, (400, ), (1, ))
    assert_size_stride(arg400_1, (1024, ), (1, ))
    assert_size_stride(arg401_1, (1024, ), (1, ))
    assert_size_stride(arg402_1, (400, ), (1, ))
    assert_size_stride(arg403_1, (400, ), (1, ))
    assert_size_stride(arg404_1, (400, ), (1, ))
    assert_size_stride(arg405_1, (400, ), (1, ))
    assert_size_stride(arg406_1, (1088, ), (1, ))
    assert_size_stride(arg407_1, (1088, ), (1, ))
    assert_size_stride(arg408_1, (400, ), (1, ))
    assert_size_stride(arg409_1, (400, ), (1, ))
    assert_size_stride(arg410_1, (400, ), (1, ))
    assert_size_stride(arg411_1, (400, ), (1, ))
    assert_size_stride(arg412_1, (1152, ), (1, ))
    assert_size_stride(arg413_1, (1152, ), (1, ))
    assert_size_stride(arg414_1, (1152, ), (1, ))
    assert_size_stride(arg415_1, (1152, ), (1, ))
    assert_size_stride(arg416_1, (800, ), (1, ))
    assert_size_stride(arg417_1, (800, ), (1, ))
    assert_size_stride(arg418_1, (800, ), (1, ))
    assert_size_stride(arg419_1, (800, ), (1, ))
    assert_size_stride(arg420_1, (1216, ), (1, ))
    assert_size_stride(arg421_1, (1216, ), (1, ))
    assert_size_stride(arg422_1, (800, ), (1, ))
    assert_size_stride(arg423_1, (800, ), (1, ))
    assert_size_stride(arg424_1, (800, ), (1, ))
    assert_size_stride(arg425_1, (800, ), (1, ))
    assert_size_stride(arg426_1, (1280, ), (1, ))
    assert_size_stride(arg427_1, (1280, ), (1, ))
    assert_size_stride(arg428_1, (800, ), (1, ))
    assert_size_stride(arg429_1, (800, ), (1, ))
    assert_size_stride(arg430_1, (800, ), (1, ))
    assert_size_stride(arg431_1, (800, ), (1, ))
    assert_size_stride(arg432_1, (1344, ), (1, ))
    assert_size_stride(arg433_1, (1344, ), (1, ))
    assert_size_stride(arg434_1, (800, ), (1, ))
    assert_size_stride(arg435_1, (800, ), (1, ))
    assert_size_stride(arg436_1, (800, ), (1, ))
    assert_size_stride(arg437_1, (800, ), (1, ))
    assert_size_stride(arg438_1, (1408, ), (1, ))
    assert_size_stride(arg439_1, (1408, ), (1, ))
    assert_size_stride(arg440_1, (800, ), (1, ))
    assert_size_stride(arg441_1, (800, ), (1, ))
    assert_size_stride(arg442_1, (800, ), (1, ))
    assert_size_stride(arg443_1, (800, ), (1, ))
    assert_size_stride(arg444_1, (1472, ), (1, ))
    assert_size_stride(arg445_1, (1472, ), (1, ))
    assert_size_stride(arg446_1, (800, ), (1, ))
    assert_size_stride(arg447_1, (800, ), (1, ))
    assert_size_stride(arg448_1, (800, ), (1, ))
    assert_size_stride(arg449_1, (800, ), (1, ))
    assert_size_stride(arg450_1, (1536, ), (1, ))
    assert_size_stride(arg451_1, (1536, ), (1, ))
    assert_size_stride(arg452_1, (800, ), (1, ))
    assert_size_stride(arg453_1, (800, ), (1, ))
    assert_size_stride(arg454_1, (800, ), (1, ))
    assert_size_stride(arg455_1, (800, ), (1, ))
    assert_size_stride(arg456_1, (1600, ), (1, ))
    assert_size_stride(arg457_1, (1600, ), (1, ))
    assert_size_stride(arg458_1, (800, ), (1, ))
    assert_size_stride(arg459_1, (800, ), (1, ))
    assert_size_stride(arg460_1, (800, ), (1, ))
    assert_size_stride(arg461_1, (800, ), (1, ))
    assert_size_stride(arg462_1, (1664, ), (1, ))
    assert_size_stride(arg463_1, (1664, ), (1, ))
    assert_size_stride(arg464_1, (800, ), (1, ))
    assert_size_stride(arg465_1, (800, ), (1, ))
    assert_size_stride(arg466_1, (800, ), (1, ))
    assert_size_stride(arg467_1, (800, ), (1, ))
    assert_size_stride(arg468_1, (1728, ), (1, ))
    assert_size_stride(arg469_1, (1728, ), (1, ))
    assert_size_stride(arg470_1, (800, ), (1, ))
    assert_size_stride(arg471_1, (800, ), (1, ))
    assert_size_stride(arg472_1, (800, ), (1, ))
    assert_size_stride(arg473_1, (800, ), (1, ))
    assert_size_stride(arg474_1, (1792, ), (1, ))
    assert_size_stride(arg475_1, (1792, ), (1, ))
    assert_size_stride(arg476_1, (800, ), (1, ))
    assert_size_stride(arg477_1, (800, ), (1, ))
    assert_size_stride(arg478_1, (800, ), (1, ))
    assert_size_stride(arg479_1, (800, ), (1, ))
    assert_size_stride(arg480_1, (1856, ), (1, ))
    assert_size_stride(arg481_1, (1856, ), (1, ))
    assert_size_stride(arg482_1, (800, ), (1, ))
    assert_size_stride(arg483_1, (800, ), (1, ))
    assert_size_stride(arg484_1, (800, ), (1, ))
    assert_size_stride(arg485_1, (800, ), (1, ))
    assert_size_stride(arg486_1, (1920, ), (1, ))
    assert_size_stride(arg487_1, (1920, ), (1, ))
    assert_size_stride(arg488_1, (800, ), (1, ))
    assert_size_stride(arg489_1, (800, ), (1, ))
    assert_size_stride(arg490_1, (800, ), (1, ))
    assert_size_stride(arg491_1, (800, ), (1, ))
    assert_size_stride(arg492_1, (1984, ), (1, ))
    assert_size_stride(arg493_1, (1984, ), (1, ))
    assert_size_stride(arg494_1, (800, ), (1, ))
    assert_size_stride(arg495_1, (800, ), (1, ))
    assert_size_stride(arg496_1, (800, ), (1, ))
    assert_size_stride(arg497_1, (800, ), (1, ))
    assert_size_stride(arg498_1, (2048, ), (1, ))
    assert_size_stride(arg499_1, (2048, ), (1, ))
    assert_size_stride(arg500_1, (800, ), (1, ))
    assert_size_stride(arg501_1, (800, ), (1, ))
    assert_size_stride(arg502_1, (800, ), (1, ))
    assert_size_stride(arg503_1, (800, ), (1, ))
    assert_size_stride(arg504_1, (2112, ), (1, ))
    assert_size_stride(arg505_1, (2112, ), (1, ))
    assert_size_stride(arg506_1, (800, ), (1, ))
    assert_size_stride(arg507_1, (800, ), (1, ))
    assert_size_stride(arg508_1, (800, ), (1, ))
    assert_size_stride(arg509_1, (800, ), (1, ))
    assert_size_stride(arg510_1, (2176, ), (1, ))
    assert_size_stride(arg511_1, (2176, ), (1, ))
    assert_size_stride(arg512_1, (800, ), (1, ))
    assert_size_stride(arg513_1, (800, ), (1, ))
    assert_size_stride(arg514_1, (800, ), (1, ))
    assert_size_stride(arg515_1, (800, ), (1, ))
    assert_size_stride(arg516_1, (2240, ), (1, ))
    assert_size_stride(arg517_1, (2240, ), (1, ))
    assert_size_stride(arg518_1, (800, ), (1, ))
    assert_size_stride(arg519_1, (800, ), (1, ))
    assert_size_stride(arg520_1, (800, ), (1, ))
    assert_size_stride(arg521_1, (800, ), (1, ))
    assert_size_stride(arg522_1, (2304, ), (1, ))
    assert_size_stride(arg523_1, (2304, ), (1, ))
    assert_size_stride(arg524_1, (800, ), (1, ))
    assert_size_stride(arg525_1, (800, ), (1, ))
    assert_size_stride(arg526_1, (800, ), (1, ))
    assert_size_stride(arg527_1, (800, ), (1, ))
    assert_size_stride(arg528_1, (2368, ), (1, ))
    assert_size_stride(arg529_1, (2368, ), (1, ))
    assert_size_stride(arg530_1, (800, ), (1, ))
    assert_size_stride(arg531_1, (800, ), (1, ))
    assert_size_stride(arg532_1, (800, ), (1, ))
    assert_size_stride(arg533_1, (800, ), (1, ))
    assert_size_stride(arg534_1, (2432, ), (1, ))
    assert_size_stride(arg535_1, (2432, ), (1, ))
    assert_size_stride(arg536_1, (2432, ), (1, ))
    assert_size_stride(arg537_1, (2432, ), (1, ))
    assert_size_stride(arg538_1, (1600, ), (1, ))
    assert_size_stride(arg539_1, (1600, ), (1, ))
    assert_size_stride(arg540_1, (1600, ), (1, ))
    assert_size_stride(arg541_1, (1600, ), (1, ))
    assert_size_stride(arg542_1, (2432, ), (1, ))
    assert_size_stride(arg543_1, (2432, ), (1, ))
    assert_size_stride(arg544_1, (1600, ), (1, ))
    assert_size_stride(arg545_1, (1600, ), (1, ))
    assert_size_stride(arg546_1, (1600, ), (1, ))
    assert_size_stride(arg547_1, (1600, ), (1, ))
    assert_size_stride(arg548_1, (2560, ), (1, ))
    assert_size_stride(arg549_1, (2560, ), (1, ))
    assert_size_stride(arg550_1, (1600, ), (1, ))
    assert_size_stride(arg551_1, (1600, ), (1, ))
    assert_size_stride(arg552_1, (1600, ), (1, ))
    assert_size_stride(arg553_1, (1600, ), (1, ))
    assert_size_stride(arg554_1, (2688, ), (1, ))
    assert_size_stride(arg555_1, (2688, ), (1, ))
    assert_size_stride(arg556_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((128, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg556_1.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg222_1
    del arg556_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 128, 112, 112), (1605632, 1, 14336, 128))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(arg337_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg338_1.data_ptr()), c_void_p(arg339_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf7.data_ptr()))
    del arg0_1
    del arg1_1
    del arg2_1
    del arg334_1
    del arg335_1
    del arg336_1
    del arg337_1
    del arg338_1
    del arg339_1
    del arg3_1
    del arg4_1
    del arg5_1
    del buf3
    del buf4
    # Source Nodes: [x_5, x_7, x_s], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf6 = extern_kernels.convolution(buf5, arg223_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf6, (8, 296, 56, 56), (928256, 1, 16576, 296))
    del arg223_1
    del buf5
    # Source Nodes: [x_10, x_8, x_in_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf8 = extern_kernels.convolution(buf7, arg224_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf8, (8, 200, 56, 56), (627200, 1, 11200, 200))
    del arg224_1
    buf9 = buf8; del buf8  # reuse
    buf10 = empty_strided((200, 4, 3, 3), (36, 1, 12, 4), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_2(c_void_p(buf9.data_ptr()), c_void_p(arg340_1.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg225_1.data_ptr()), c_void_p(buf10.data_ptr()))
    del arg225_1
    del arg340_1
    del arg341_1
    del arg6_1
    del arg7_1
    # Source Nodes: [x_11, x_13, x_in_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf11 = extern_kernels.convolution(buf9, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf11, (8, 200, 56, 56), (627200, 1, 11200, 200))
    del buf9
    buf12 = buf11; del buf11  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_3(c_void_p(buf12.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(arg343_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()))
    del arg342_1
    del arg343_1
    del arg8_1
    del arg9_1
    # Source Nodes: [x_14, x_16, x_in_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf13 = extern_kernels.convolution(buf12, arg226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf13, (8, 276, 56, 56), (865536, 1, 15456, 276))
    del arg226_1
    del buf12
    buf14 = empty_strided((8, 316, 56, 56), (990976, 1, 17696, 316), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_4(c_void_p(buf6.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(arg344_1.data_ptr()), c_void_p(arg345_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf14.data_ptr()))
    del arg10_1
    del arg11_1
    del arg344_1
    del arg345_1
    # Source Nodes: [cat_138, x_17, x_19, x_in_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
    buf15 = extern_kernels.convolution(buf14, arg227_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf15, (8, 200, 56, 56), (627200, 1, 11200, 200))
    del arg227_1
    del buf14
    buf16 = buf15; del buf15  # reuse
    buf17 = buf10; del buf10  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_5(c_void_p(buf16.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(arg347_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(buf17.data_ptr()))
    del arg12_1
    del arg13_1
    del arg228_1
    del arg346_1
    del arg347_1
    # Source Nodes: [x_20, x_22, x_in_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf18 = extern_kernels.convolution(buf16, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf18, (8, 200, 56, 56), (627200, 1, 11200, 200))
    del buf16
    buf19 = buf18; del buf18  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_6(c_void_p(buf19.data_ptr()), c_void_p(arg348_1.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()))
    del arg14_1
    del arg15_1
    del arg348_1
    del arg349_1
    # Source Nodes: [x_23, x_25, x_in_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf20 = extern_kernels.convolution(buf19, arg229_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf20, (8, 276, 56, 56), (865536, 1, 15456, 276))
    del arg229_1
    del buf19
    buf21 = empty_strided((8, 336, 56, 56), (1053696, 1, 18816, 336), device='cpu', dtype=torch.float32)
    buf22 = buf21; del buf21  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_7(c_void_p(buf22.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(arg350_1.data_ptr()), c_void_p(arg351_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()))
    del arg16_1
    del arg17_1
    del arg350_1
    del arg351_1
    # Source Nodes: [x_26, x_28, x_in_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf23 = extern_kernels.convolution(buf22, arg230_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf23, (8, 200, 56, 56), (627200, 1, 11200, 200))
    del arg230_1
    del buf22
    buf24 = buf23; del buf23  # reuse
    buf25 = buf17; del buf17  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_8(c_void_p(buf24.data_ptr()), c_void_p(arg352_1.data_ptr()), c_void_p(arg353_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(buf25.data_ptr()))
    del arg18_1
    del arg19_1
    del arg231_1
    del arg352_1
    del arg353_1
    # Source Nodes: [x_29, x_31, x_in_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf26 = extern_kernels.convolution(buf24, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf26, (8, 200, 56, 56), (627200, 1, 11200, 200))
    del buf24
    buf27 = buf26; del buf26  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_9(c_void_p(buf27.data_ptr()), c_void_p(arg354_1.data_ptr()), c_void_p(arg355_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()))
    del arg20_1
    del arg21_1
    del arg354_1
    del arg355_1
    # Source Nodes: [x_32, x_34, x_in_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf28 = extern_kernels.convolution(buf27, arg232_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf28, (8, 276, 56, 56), (865536, 1, 15456, 276))
    del arg232_1
    del buf27
    buf29 = empty_strided((8, 356, 56, 56), (1116416, 1, 19936, 356), device='cpu', dtype=torch.float32)
    buf30 = buf29; del buf29  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_10(c_void_p(buf30.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(arg356_1.data_ptr()), c_void_p(arg357_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()))
    del arg22_1
    del arg23_1
    del arg356_1
    del arg357_1
    # Source Nodes: [x_35, x_37, x_in_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf31 = extern_kernels.convolution(buf30, arg233_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf31, (8, 200, 56, 56), (627200, 1, 11200, 200))
    del arg233_1
    del buf30
    buf32 = buf31; del buf31  # reuse
    buf33 = buf25; del buf25  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_11(c_void_p(buf32.data_ptr()), c_void_p(arg358_1.data_ptr()), c_void_p(arg359_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(buf33.data_ptr()))
    del arg234_1
    del arg24_1
    del arg25_1
    del arg358_1
    del arg359_1
    # Source Nodes: [x_38, x_40, x_in_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf34 = extern_kernels.convolution(buf32, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf34, (8, 200, 56, 56), (627200, 1, 11200, 200))
    del buf32
    del buf33
    buf35 = buf34; del buf34  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_12(c_void_p(buf35.data_ptr()), c_void_p(arg360_1.data_ptr()), c_void_p(arg361_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()))
    del arg26_1
    del arg27_1
    del arg360_1
    del arg361_1
    # Source Nodes: [x_41, x_43, x_in_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf36 = extern_kernels.convolution(buf35, arg235_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf36, (8, 276, 56, 56), (865536, 1, 15456, 276))
    del arg235_1
    del buf35
    buf37 = empty_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cpu', dtype=torch.float32)
    buf39 = empty_strided((8, 376, 56, 56), (1179136, 1, 21056, 376), device='cpu', dtype=torch.float32)
    buf41 = empty_strided((8, 376, 56, 56), (1179136, 1, 21056, 376), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_13(c_void_p(buf6.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(arg362_1.data_ptr()), c_void_p(arg363_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(arg364_1.data_ptr()), c_void_p(arg365_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf41.data_ptr()))
    del arg28_1
    del arg29_1
    del arg30_1
    del arg31_1
    del arg362_1
    del arg363_1
    del arg364_1
    del arg365_1
    del buf13
    del buf20
    del buf28
    del buf36
    del buf6
    # Source Nodes: [x_44, x_46, x_s_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf40 = extern_kernels.convolution(buf39, arg236_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf40, (8, 640, 28, 28), (501760, 1, 17920, 640))
    del arg236_1
    del buf39
    # Source Nodes: [x_47, x_49, x_in_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf42 = extern_kernels.convolution(buf41, arg237_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf42, (8, 400, 56, 56), (1254400, 1, 22400, 400))
    del arg237_1
    del buf41
    buf43 = buf42; del buf42  # reuse
    buf44 = empty_strided((400, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_14(c_void_p(buf43.data_ptr()), c_void_p(arg366_1.data_ptr()), c_void_p(arg367_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(buf44.data_ptr()))
    del arg238_1
    del arg32_1
    del arg33_1
    del arg366_1
    del arg367_1
    # Source Nodes: [x_50, x_52, x_in_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf45 = extern_kernels.convolution(buf43, buf44, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf45, (8, 400, 28, 28), (313600, 1, 11200, 400))
    del buf43
    buf46 = buf45; del buf45  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_15(c_void_p(buf46.data_ptr()), c_void_p(arg368_1.data_ptr()), c_void_p(arg369_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()))
    del arg34_1
    del arg35_1
    del arg368_1
    del arg369_1
    # Source Nodes: [x_53, x_55, x_in_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf47 = extern_kernels.convolution(buf46, arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf47, (8, 576, 28, 28), (451584, 1, 16128, 576))
    del arg239_1
    del buf46
    buf48 = empty_strided((8, 704, 28, 28), (551936, 1, 19712, 704), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_16(c_void_p(buf40.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(arg370_1.data_ptr()), c_void_p(arg371_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(buf48.data_ptr()))
    del arg36_1
    del arg370_1
    del arg371_1
    del arg37_1
    # Source Nodes: [cat_130, x_56, x_58, x_in_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
    buf49 = extern_kernels.convolution(buf48, arg240_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf49, (8, 400, 28, 28), (313600, 1, 11200, 400))
    del arg240_1
    del buf48
    buf50 = buf49; del buf49  # reuse
    buf51 = buf44; del buf44  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_17(c_void_p(buf50.data_ptr()), c_void_p(arg372_1.data_ptr()), c_void_p(arg373_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(buf51.data_ptr()))
    del arg241_1
    del arg372_1
    del arg373_1
    del arg38_1
    del arg39_1
    # Source Nodes: [x_59, x_61, x_in_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf52 = extern_kernels.convolution(buf50, buf51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf52, (8, 400, 28, 28), (313600, 1, 11200, 400))
    del buf50
    buf53 = buf52; del buf52  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_18(c_void_p(buf53.data_ptr()), c_void_p(arg374_1.data_ptr()), c_void_p(arg375_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()))
    del arg374_1
    del arg375_1
    del arg40_1
    del arg41_1
    # Source Nodes: [x_62, x_64, x_in_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf54 = extern_kernels.convolution(buf53, arg242_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf54, (8, 576, 28, 28), (451584, 1, 16128, 576))
    del arg242_1
    del buf53
    buf55 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cpu', dtype=torch.float32)
    buf56 = buf55; del buf55  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_19(c_void_p(buf56.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(arg376_1.data_ptr()), c_void_p(arg377_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()))
    del arg376_1
    del arg377_1
    del arg42_1
    del arg43_1
    # Source Nodes: [x_65, x_67, x_in_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf57 = extern_kernels.convolution(buf56, arg243_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf57, (8, 400, 28, 28), (313600, 1, 11200, 400))
    del arg243_1
    del buf56
    buf58 = buf57; del buf57  # reuse
    buf59 = buf51; del buf51  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_20(c_void_p(buf58.data_ptr()), c_void_p(arg378_1.data_ptr()), c_void_p(arg379_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(buf59.data_ptr()))
    del arg244_1
    del arg378_1
    del arg379_1
    del arg44_1
    del arg45_1
    # Source Nodes: [x_68, x_70, x_in_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf60 = extern_kernels.convolution(buf58, buf59, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf60, (8, 400, 28, 28), (313600, 1, 11200, 400))
    del buf58
    buf61 = buf60; del buf60  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_21(c_void_p(buf61.data_ptr()), c_void_p(arg380_1.data_ptr()), c_void_p(arg381_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()))
    del arg380_1
    del arg381_1
    del arg46_1
    del arg47_1
    # Source Nodes: [x_71, x_73, x_in_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf62 = extern_kernels.convolution(buf61, arg245_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf62, (8, 576, 28, 28), (451584, 1, 16128, 576))
    del arg245_1
    del buf61
    buf63 = empty_strided((8, 832, 28, 28), (652288, 1, 23296, 832), device='cpu', dtype=torch.float32)
    buf64 = buf63; del buf63  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_22(c_void_p(buf64.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(arg382_1.data_ptr()), c_void_p(arg383_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()))
    del arg382_1
    del arg383_1
    del arg48_1
    del arg49_1
    # Source Nodes: [x_74, x_76, x_in_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf65 = extern_kernels.convolution(buf64, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf65, (8, 400, 28, 28), (313600, 1, 11200, 400))
    del arg246_1
    del buf64
    buf66 = buf65; del buf65  # reuse
    buf67 = buf59; del buf59  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_23(c_void_p(buf66.data_ptr()), c_void_p(arg384_1.data_ptr()), c_void_p(arg385_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(buf67.data_ptr()))
    del arg247_1
    del arg384_1
    del arg385_1
    del arg50_1
    del arg51_1
    # Source Nodes: [x_77, x_79, x_in_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf68 = extern_kernels.convolution(buf66, buf67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf68, (8, 400, 28, 28), (313600, 1, 11200, 400))
    del buf66
    buf69 = buf68; del buf68  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_24(c_void_p(buf69.data_ptr()), c_void_p(arg386_1.data_ptr()), c_void_p(arg387_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()))
    del arg386_1
    del arg387_1
    del arg52_1
    del arg53_1
    # Source Nodes: [x_80, x_82, x_in_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf70 = extern_kernels.convolution(buf69, arg248_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf70, (8, 576, 28, 28), (451584, 1, 16128, 576))
    del arg248_1
    del buf69
    buf73 = empty((8, 896, 28, 28), device='cpu', dtype=torch.float32)
    buf71 = reinterpret_tensor(buf73, (8, 512, 28, 28), (702464, 784, 28, 1), 0)  # alias
    buf72 = reinterpret_tensor(buf73, (8, 384, 28, 28), (702464, 784, 28, 1), 401408)  # alias
    buf74 = empty_strided((8, 896, 28, 28), (702464, 1, 25088, 896), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_cat_relu_25(c_void_p(buf40.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(arg388_1.data_ptr()), c_void_p(arg389_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf74.data_ptr()))
    del arg388_1
    del arg389_1
    del arg54_1
    del arg55_1
    del buf47
    del buf54
    del buf62
    del buf70
    # Source Nodes: [x_83, x_85, x_in_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf75 = extern_kernels.convolution(buf74, arg249_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf75, (8, 400, 28, 28), (313600, 1, 11200, 400))
    del arg249_1
    del buf74
    buf76 = buf75; del buf75  # reuse
    buf77 = buf67; del buf67  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_26(c_void_p(buf76.data_ptr()), c_void_p(arg390_1.data_ptr()), c_void_p(arg391_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(buf77.data_ptr()))
    del arg250_1
    del arg390_1
    del arg391_1
    del arg56_1
    del arg57_1
    # Source Nodes: [x_86, x_88, x_in_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf78 = extern_kernels.convolution(buf76, buf77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf78, (8, 400, 28, 28), (313600, 1, 11200, 400))
    del buf76
    buf79 = buf78; del buf78  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_27(c_void_p(buf79.data_ptr()), c_void_p(arg392_1.data_ptr()), c_void_p(arg393_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()))
    del arg392_1
    del arg393_1
    del arg58_1
    del arg59_1
    # Source Nodes: [x_89, x_91, x_in_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf80 = extern_kernels.convolution(buf79, arg251_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf80, (8, 576, 28, 28), (451584, 1, 16128, 576))
    del arg251_1
    del buf79
    buf81 = empty_strided((8, 960, 28, 28), (752640, 1, 26880, 960), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_28(c_void_p(buf71.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(arg394_1.data_ptr()), c_void_p(arg395_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(buf81.data_ptr()))
    del arg394_1
    del arg395_1
    del arg60_1
    del arg61_1
    # Source Nodes: [cat_122, x_92, x_94, x_in_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
    buf82 = extern_kernels.convolution(buf81, arg252_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf82, (8, 400, 28, 28), (313600, 1, 11200, 400))
    del arg252_1
    del buf81
    buf83 = buf82; del buf82  # reuse
    buf84 = buf77; del buf77  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_29(c_void_p(buf83.data_ptr()), c_void_p(arg396_1.data_ptr()), c_void_p(arg397_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(buf84.data_ptr()))
    del arg253_1
    del arg396_1
    del arg397_1
    del arg62_1
    del arg63_1
    # Source Nodes: [x_95, x_97, x_in_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf85 = extern_kernels.convolution(buf83, buf84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf85, (8, 400, 28, 28), (313600, 1, 11200, 400))
    del buf83
    buf86 = buf85; del buf85  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_30(c_void_p(buf86.data_ptr()), c_void_p(arg398_1.data_ptr()), c_void_p(arg399_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()))
    del arg398_1
    del arg399_1
    del arg64_1
    del arg65_1
    # Source Nodes: [x_100, x_98, x_in_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf87 = extern_kernels.convolution(buf86, arg254_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf87, (8, 576, 28, 28), (451584, 1, 16128, 576))
    del arg254_1
    del buf86
    buf88 = empty_strided((8, 1024, 28, 28), (802816, 1, 28672, 1024), device='cpu', dtype=torch.float32)
    buf89 = buf88; del buf88  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_31(c_void_p(buf89.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(arg400_1.data_ptr()), c_void_p(arg401_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()))
    del arg400_1
    del arg401_1
    del arg66_1
    del arg67_1
    # Source Nodes: [x_101, x_103, x_in_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf90 = extern_kernels.convolution(buf89, arg255_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf90, (8, 400, 28, 28), (313600, 1, 11200, 400))
    del arg255_1
    del buf89
    buf91 = buf90; del buf90  # reuse
    buf92 = buf84; del buf84  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_32(c_void_p(buf91.data_ptr()), c_void_p(arg402_1.data_ptr()), c_void_p(arg403_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(buf92.data_ptr()))
    del arg256_1
    del arg402_1
    del arg403_1
    del arg68_1
    del arg69_1
    # Source Nodes: [x_104, x_106, x_in_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf93 = extern_kernels.convolution(buf91, buf92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf93, (8, 400, 28, 28), (313600, 1, 11200, 400))
    del buf91
    buf94 = buf93; del buf93  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_33(c_void_p(buf94.data_ptr()), c_void_p(arg404_1.data_ptr()), c_void_p(arg405_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()))
    del arg404_1
    del arg405_1
    del arg70_1
    del arg71_1
    # Source Nodes: [x_107, x_109, x_in_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf95 = extern_kernels.convolution(buf94, arg257_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf95, (8, 576, 28, 28), (451584, 1, 16128, 576))
    del arg257_1
    del buf94
    buf96 = empty_strided((8, 1088, 28, 28), (852992, 1, 30464, 1088), device='cpu', dtype=torch.float32)
    buf97 = buf96; del buf96  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_34(c_void_p(buf97.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(arg406_1.data_ptr()), c_void_p(arg407_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()))
    del arg406_1
    del arg407_1
    del arg72_1
    del arg73_1
    # Source Nodes: [x_110, x_112, x_in_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf98 = extern_kernels.convolution(buf97, arg258_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf98, (8, 400, 28, 28), (313600, 1, 11200, 400))
    del arg258_1
    del buf97
    buf99 = buf98; del buf98  # reuse
    buf100 = buf92; del buf92  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_35(c_void_p(buf99.data_ptr()), c_void_p(arg408_1.data_ptr()), c_void_p(arg409_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(buf100.data_ptr()))
    del arg259_1
    del arg408_1
    del arg409_1
    del arg74_1
    del arg75_1
    # Source Nodes: [x_113, x_115, x_in_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf101 = extern_kernels.convolution(buf99, buf100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf101, (8, 400, 28, 28), (313600, 1, 11200, 400))
    del buf100
    del buf99
    buf102 = buf101; del buf101  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_36(c_void_p(buf102.data_ptr()), c_void_p(arg410_1.data_ptr()), c_void_p(arg411_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()))
    del arg410_1
    del arg411_1
    del arg76_1
    del arg77_1
    # Source Nodes: [x_116, x_118, x_in_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf103 = extern_kernels.convolution(buf102, arg260_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf103, (8, 576, 28, 28), (451584, 1, 16128, 576))
    del arg260_1
    buf104 = buf40; del buf40  # reuse
    buf105 = empty_strided((8, 1152, 28, 28), (903168, 1, 32256, 1152), device='cpu', dtype=torch.float32)
    buf106 = empty_strided((8, 1152, 28, 28), (903168, 1, 32256, 1152), device='cpu', dtype=torch.float32)
    buf108 = empty_strided((8, 1152, 28, 28), (903168, 1, 32256, 1152), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_37(c_void_p(buf72.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(arg412_1.data_ptr()), c_void_p(arg413_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(arg414_1.data_ptr()), c_void_p(arg415_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf108.data_ptr()))
    del arg412_1
    del arg413_1
    del arg414_1
    del arg415_1
    del arg78_1
    del arg79_1
    del arg80_1
    del arg81_1
    del buf103
    del buf104
    del buf105
    del buf71
    del buf72
    del buf73
    del buf80
    del buf87
    # Source Nodes: [x_119, x_121, x_s_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf107 = extern_kernels.convolution(buf106, arg261_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf107, (8, 1152, 14, 14), (225792, 1, 16128, 1152))
    del arg261_1
    del buf106
    # Source Nodes: [x_122, x_124, x_in_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf109 = extern_kernels.convolution(buf108, arg262_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf109, (8, 800, 28, 28), (627200, 1, 22400, 800))
    del arg262_1
    del buf108
    buf110 = buf109; del buf109  # reuse
    buf111 = empty_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_38(c_void_p(buf110.data_ptr()), c_void_p(arg416_1.data_ptr()), c_void_p(arg417_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(buf111.data_ptr()))
    del arg263_1
    del arg416_1
    del arg417_1
    del arg82_1
    del arg83_1
    # Source Nodes: [x_125, x_127, x_in_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf112 = extern_kernels.convolution(buf110, buf111, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf112, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf110
    buf113 = buf112; del buf112  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_39(c_void_p(buf113.data_ptr()), c_void_p(arg418_1.data_ptr()), c_void_p(arg419_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(arg85_1.data_ptr()))
    del arg418_1
    del arg419_1
    del arg84_1
    del arg85_1
    # Source Nodes: [x_128, x_130, x_in_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf114 = extern_kernels.convolution(buf113, arg264_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf114, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg264_1
    del buf113
    buf115 = empty_strided((8, 1216, 14, 14), (238336, 1, 17024, 1216), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_40(c_void_p(buf107.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(arg420_1.data_ptr()), c_void_p(arg421_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(buf115.data_ptr()))
    del arg420_1
    del arg421_1
    del arg86_1
    del arg87_1
    # Source Nodes: [cat_114, x_131, x_133, x_in_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
    buf116 = extern_kernels.convolution(buf115, arg265_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf116, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg265_1
    del buf115
    buf117 = buf116; del buf116  # reuse
    buf118 = buf111; del buf111  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_41(c_void_p(buf117.data_ptr()), c_void_p(arg422_1.data_ptr()), c_void_p(arg423_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(buf118.data_ptr()))
    del arg266_1
    del arg422_1
    del arg423_1
    del arg88_1
    del arg89_1
    # Source Nodes: [x_134, x_136, x_in_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf119 = extern_kernels.convolution(buf117, buf118, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf119, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf117
    buf120 = buf119; del buf119  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_42(c_void_p(buf120.data_ptr()), c_void_p(arg424_1.data_ptr()), c_void_p(arg425_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(arg91_1.data_ptr()))
    del arg424_1
    del arg425_1
    del arg90_1
    del arg91_1
    # Source Nodes: [x_137, x_139, x_in_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf121 = extern_kernels.convolution(buf120, arg267_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf121, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg267_1
    del buf120
    buf122 = empty_strided((8, 1280, 14, 14), (250880, 1, 17920, 1280), device='cpu', dtype=torch.float32)
    buf123 = buf122; del buf122  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_43(c_void_p(buf123.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(arg426_1.data_ptr()), c_void_p(arg427_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()))
    del arg426_1
    del arg427_1
    del arg92_1
    del arg93_1
    # Source Nodes: [x_140, x_142, x_in_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf124 = extern_kernels.convolution(buf123, arg268_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf124, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg268_1
    del buf123
    buf125 = buf124; del buf124  # reuse
    buf126 = buf118; del buf118  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_44(c_void_p(buf125.data_ptr()), c_void_p(arg428_1.data_ptr()), c_void_p(arg429_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(buf126.data_ptr()))
    del arg269_1
    del arg428_1
    del arg429_1
    del arg94_1
    del arg95_1
    # Source Nodes: [x_143, x_145, x_in_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf127 = extern_kernels.convolution(buf125, buf126, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf127, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf125
    buf128 = buf127; del buf127  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_45(c_void_p(buf128.data_ptr()), c_void_p(arg430_1.data_ptr()), c_void_p(arg431_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()))
    del arg430_1
    del arg431_1
    del arg96_1
    del arg97_1
    # Source Nodes: [x_146, x_148, x_in_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf129 = extern_kernels.convolution(buf128, arg270_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf129, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg270_1
    del buf128
    buf130 = empty_strided((8, 1344, 14, 14), (263424, 1, 18816, 1344), device='cpu', dtype=torch.float32)
    buf131 = buf130; del buf130  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_46(c_void_p(buf131.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(arg432_1.data_ptr()), c_void_p(arg433_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()))
    del arg432_1
    del arg433_1
    del arg98_1
    del arg99_1
    # Source Nodes: [x_149, x_151, x_in_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf132 = extern_kernels.convolution(buf131, arg271_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf132, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg271_1
    del buf131
    buf133 = buf132; del buf132  # reuse
    buf134 = buf126; del buf126  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_47(c_void_p(buf133.data_ptr()), c_void_p(arg434_1.data_ptr()), c_void_p(arg435_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(buf134.data_ptr()))
    del arg100_1
    del arg101_1
    del arg272_1
    del arg434_1
    del arg435_1
    # Source Nodes: [x_152, x_154, x_in_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf135 = extern_kernels.convolution(buf133, buf134, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf135, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf133
    buf136 = buf135; del buf135  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_48(c_void_p(buf136.data_ptr()), c_void_p(arg436_1.data_ptr()), c_void_p(arg437_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()))
    del arg102_1
    del arg103_1
    del arg436_1
    del arg437_1
    # Source Nodes: [x_155, x_157, x_in_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf137 = extern_kernels.convolution(buf136, arg273_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf137, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg273_1
    del buf136
    buf140 = empty((8, 1408, 14, 14), device='cpu', dtype=torch.float32)
    buf138 = reinterpret_tensor(buf140, (8, 1024, 14, 14), (275968, 196, 14, 1), 0)  # alias
    buf139 = reinterpret_tensor(buf140, (8, 384, 14, 14), (275968, 196, 14, 1), 200704)  # alias
    buf141 = empty_strided((8, 1408, 14, 14), (275968, 1, 19712, 1408), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_cat_relu_49(c_void_p(buf107.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(arg438_1.data_ptr()), c_void_p(arg439_1.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf141.data_ptr()))
    del arg104_1
    del arg105_1
    del arg438_1
    del arg439_1
    del buf107
    del buf114
    del buf121
    del buf129
    del buf137
    # Source Nodes: [x_158, x_160, x_in_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf142 = extern_kernels.convolution(buf141, arg274_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf142, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg274_1
    del buf141
    buf143 = buf142; del buf142  # reuse
    buf144 = buf134; del buf134  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_50(c_void_p(buf143.data_ptr()), c_void_p(arg440_1.data_ptr()), c_void_p(arg441_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(buf144.data_ptr()))
    del arg106_1
    del arg107_1
    del arg275_1
    del arg440_1
    del arg441_1
    # Source Nodes: [x_161, x_163, x_in_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf145 = extern_kernels.convolution(buf143, buf144, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf145, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf143
    buf146 = buf145; del buf145  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_51(c_void_p(buf146.data_ptr()), c_void_p(arg442_1.data_ptr()), c_void_p(arg443_1.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg109_1.data_ptr()))
    del arg108_1
    del arg109_1
    del arg442_1
    del arg443_1
    # Source Nodes: [x_164, x_166, x_in_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf147 = extern_kernels.convolution(buf146, arg276_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf147, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg276_1
    del buf146
    buf148 = empty_strided((8, 1472, 14, 14), (288512, 1, 20608, 1472), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_52(c_void_p(buf138.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(arg444_1.data_ptr()), c_void_p(arg445_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(buf148.data_ptr()))
    del arg110_1
    del arg111_1
    del arg444_1
    del arg445_1
    # Source Nodes: [cat_106, x_167, x_169, x_in_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
    buf149 = extern_kernels.convolution(buf148, arg277_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf149, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg277_1
    del buf148
    buf150 = buf149; del buf149  # reuse
    buf151 = buf144; del buf144  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_53(c_void_p(buf150.data_ptr()), c_void_p(arg446_1.data_ptr()), c_void_p(arg447_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(arg278_1.data_ptr()), c_void_p(buf151.data_ptr()))
    del arg112_1
    del arg113_1
    del arg278_1
    del arg446_1
    del arg447_1
    # Source Nodes: [x_170, x_172, x_in_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf152 = extern_kernels.convolution(buf150, buf151, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf152, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf150
    buf153 = buf152; del buf152  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_54(c_void_p(buf153.data_ptr()), c_void_p(arg448_1.data_ptr()), c_void_p(arg449_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg115_1.data_ptr()))
    del arg114_1
    del arg115_1
    del arg448_1
    del arg449_1
    # Source Nodes: [x_173, x_175, x_in_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf154 = extern_kernels.convolution(buf153, arg279_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf154, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg279_1
    del buf153
    buf155 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    buf156 = buf155; del buf155  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_55(c_void_p(buf156.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(arg450_1.data_ptr()), c_void_p(arg451_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg117_1.data_ptr()))
    del arg116_1
    del arg117_1
    del arg450_1
    del arg451_1
    # Source Nodes: [x_176, x_178, x_in_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf157 = extern_kernels.convolution(buf156, arg280_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf157, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg280_1
    del buf156
    buf158 = buf157; del buf157  # reuse
    buf159 = buf151; del buf151  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_56(c_void_p(buf158.data_ptr()), c_void_p(arg452_1.data_ptr()), c_void_p(arg453_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(buf159.data_ptr()))
    del arg118_1
    del arg119_1
    del arg281_1
    del arg452_1
    del arg453_1
    # Source Nodes: [x_179, x_181, x_in_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf160 = extern_kernels.convolution(buf158, buf159, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf160, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf158
    buf161 = buf160; del buf160  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_57(c_void_p(buf161.data_ptr()), c_void_p(arg454_1.data_ptr()), c_void_p(arg455_1.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(arg121_1.data_ptr()))
    del arg120_1
    del arg121_1
    del arg454_1
    del arg455_1
    # Source Nodes: [x_182, x_184, x_in_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf162 = extern_kernels.convolution(buf161, arg282_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf162, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg282_1
    del buf161
    buf163 = reinterpret_tensor(buf102, (8, 1600, 14, 14), (313600, 1, 22400, 1600), 0); del buf102  # reuse
    buf164 = buf163; del buf163  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_58(c_void_p(buf164.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(arg456_1.data_ptr()), c_void_p(arg457_1.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()))
    del arg122_1
    del arg123_1
    del arg456_1
    del arg457_1
    # Source Nodes: [x_185, x_187, x_in_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf165 = extern_kernels.convolution(buf164, arg283_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf165, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg283_1
    del buf164
    buf166 = buf165; del buf165  # reuse
    buf167 = buf159; del buf159  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_59(c_void_p(buf166.data_ptr()), c_void_p(arg458_1.data_ptr()), c_void_p(arg459_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(buf167.data_ptr()))
    del arg124_1
    del arg125_1
    del arg284_1
    del arg458_1
    del arg459_1
    # Source Nodes: [x_188, x_190, x_in_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf168 = extern_kernels.convolution(buf166, buf167, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf168, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf166
    buf169 = buf168; del buf168  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_60(c_void_p(buf169.data_ptr()), c_void_p(arg460_1.data_ptr()), c_void_p(arg461_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg127_1.data_ptr()))
    del arg126_1
    del arg127_1
    del arg460_1
    del arg461_1
    # Source Nodes: [x_191, x_193, x_in_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf170 = extern_kernels.convolution(buf169, arg285_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf170, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg285_1
    del buf169
    buf173 = empty((8, 1664, 14, 14), device='cpu', dtype=torch.float32)
    buf171 = reinterpret_tensor(buf173, (8, 1024, 14, 14), (326144, 196, 14, 1), 0)  # alias
    buf172 = reinterpret_tensor(buf173, (8, 640, 14, 14), (326144, 196, 14, 1), 200704)  # alias
    buf174 = empty_strided((8, 1664, 14, 14), (326144, 1, 23296, 1664), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_cat_relu_61(c_void_p(buf138.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(arg462_1.data_ptr()), c_void_p(arg463_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(arg129_1.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf174.data_ptr()))
    del arg128_1
    del arg129_1
    del arg462_1
    del arg463_1
    del buf138
    del buf139
    del buf147
    del buf154
    del buf162
    del buf170
    # Source Nodes: [x_194, x_196, x_in_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf175 = extern_kernels.convolution(buf174, arg286_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf175, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg286_1
    del buf174
    buf176 = buf175; del buf175  # reuse
    buf177 = buf167; del buf167  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_62(c_void_p(buf176.data_ptr()), c_void_p(arg464_1.data_ptr()), c_void_p(arg465_1.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(buf177.data_ptr()))
    del arg130_1
    del arg131_1
    del arg287_1
    del arg464_1
    del arg465_1
    # Source Nodes: [x_197, x_199, x_in_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf178 = extern_kernels.convolution(buf176, buf177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf178, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf176
    buf179 = buf178; del buf178  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_63(c_void_p(buf179.data_ptr()), c_void_p(arg466_1.data_ptr()), c_void_p(arg467_1.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg133_1.data_ptr()))
    del arg132_1
    del arg133_1
    del arg466_1
    del arg467_1
    # Source Nodes: [x_200, x_202, x_in_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf180 = extern_kernels.convolution(buf179, arg288_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf180, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg288_1
    del buf179
    buf181 = empty_strided((8, 1728, 14, 14), (338688, 1, 24192, 1728), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_64(c_void_p(buf171.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(arg468_1.data_ptr()), c_void_p(arg469_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(arg135_1.data_ptr()), c_void_p(buf181.data_ptr()))
    del arg134_1
    del arg135_1
    del arg468_1
    del arg469_1
    # Source Nodes: [cat_98, x_203, x_205, x_in_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
    buf182 = extern_kernels.convolution(buf181, arg289_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf182, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg289_1
    del buf181
    buf183 = buf182; del buf182  # reuse
    buf184 = buf177; del buf177  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_65(c_void_p(buf183.data_ptr()), c_void_p(arg470_1.data_ptr()), c_void_p(arg471_1.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(arg137_1.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(buf184.data_ptr()))
    del arg136_1
    del arg137_1
    del arg290_1
    del arg470_1
    del arg471_1
    # Source Nodes: [x_206, x_208, x_in_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf185 = extern_kernels.convolution(buf183, buf184, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf185, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf183
    buf186 = buf185; del buf185  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_66(c_void_p(buf186.data_ptr()), c_void_p(arg472_1.data_ptr()), c_void_p(arg473_1.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(arg139_1.data_ptr()))
    del arg138_1
    del arg139_1
    del arg472_1
    del arg473_1
    # Source Nodes: [x_209, x_211, x_in_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf187 = extern_kernels.convolution(buf186, arg291_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf187, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg291_1
    del buf186
    buf188 = empty_strided((8, 1792, 14, 14), (351232, 1, 25088, 1792), device='cpu', dtype=torch.float32)
    buf189 = buf188; del buf188  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_67(c_void_p(buf189.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(arg474_1.data_ptr()), c_void_p(arg475_1.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(arg141_1.data_ptr()))
    del arg140_1
    del arg141_1
    del arg474_1
    del arg475_1
    # Source Nodes: [x_212, x_214, x_in_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf190 = extern_kernels.convolution(buf189, arg292_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf190, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg292_1
    del buf189
    buf191 = buf190; del buf190  # reuse
    buf192 = buf184; del buf184  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_68(c_void_p(buf191.data_ptr()), c_void_p(arg476_1.data_ptr()), c_void_p(arg477_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg143_1.data_ptr()), c_void_p(arg293_1.data_ptr()), c_void_p(buf192.data_ptr()))
    del arg142_1
    del arg143_1
    del arg293_1
    del arg476_1
    del arg477_1
    # Source Nodes: [x_215, x_217, x_in_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf193 = extern_kernels.convolution(buf191, buf192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf193, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf191
    buf194 = buf193; del buf193  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_69(c_void_p(buf194.data_ptr()), c_void_p(arg478_1.data_ptr()), c_void_p(arg479_1.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(arg145_1.data_ptr()))
    del arg144_1
    del arg145_1
    del arg478_1
    del arg479_1
    # Source Nodes: [x_218, x_220, x_in_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf195 = extern_kernels.convolution(buf194, arg294_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf195, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg294_1
    del buf194
    buf196 = empty_strided((8, 1856, 14, 14), (363776, 1, 25984, 1856), device='cpu', dtype=torch.float32)
    buf197 = buf196; del buf196  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_70(c_void_p(buf197.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(arg480_1.data_ptr()), c_void_p(arg481_1.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(arg147_1.data_ptr()))
    del arg146_1
    del arg147_1
    del arg480_1
    del arg481_1
    # Source Nodes: [x_221, x_223, x_in_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf198 = extern_kernels.convolution(buf197, arg295_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf198, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg295_1
    del buf197
    buf199 = buf198; del buf198  # reuse
    buf200 = buf192; del buf192  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_71(c_void_p(buf199.data_ptr()), c_void_p(arg482_1.data_ptr()), c_void_p(arg483_1.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(arg149_1.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(buf200.data_ptr()))
    del arg148_1
    del arg149_1
    del arg296_1
    del arg482_1
    del arg483_1
    # Source Nodes: [x_224, x_226, x_in_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf201 = extern_kernels.convolution(buf199, buf200, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf201, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf199
    buf202 = buf201; del buf201  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_72(c_void_p(buf202.data_ptr()), c_void_p(arg484_1.data_ptr()), c_void_p(arg485_1.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(arg151_1.data_ptr()))
    del arg150_1
    del arg151_1
    del arg484_1
    del arg485_1
    # Source Nodes: [x_227, x_229, x_in_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf203 = extern_kernels.convolution(buf202, arg297_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf203, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg297_1
    del buf202
    buf206 = reinterpret_tensor(buf37, (8, 1920, 14, 14), (376320, 196, 14, 1), 0); del buf37  # reuse
    buf204 = reinterpret_tensor(buf206, (8, 1024, 14, 14), (376320, 196, 14, 1), 0)  # alias
    buf205 = reinterpret_tensor(buf206, (8, 896, 14, 14), (376320, 196, 14, 1), 200704)  # alias
    buf207 = empty_strided((8, 1920, 14, 14), (376320, 1, 26880, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_cat_relu_73(c_void_p(buf171.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(arg486_1.data_ptr()), c_void_p(arg487_1.data_ptr()), c_void_p(arg152_1.data_ptr()), c_void_p(arg153_1.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf207.data_ptr()))
    del arg152_1
    del arg153_1
    del arg486_1
    del arg487_1
    del buf171
    del buf172
    del buf173
    del buf180
    del buf187
    del buf195
    del buf203
    # Source Nodes: [x_230, x_232, x_in_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf208 = extern_kernels.convolution(buf207, arg298_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf208, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg298_1
    del buf207
    buf209 = buf208; del buf208  # reuse
    buf210 = buf200; del buf200  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_74(c_void_p(buf209.data_ptr()), c_void_p(arg488_1.data_ptr()), c_void_p(arg489_1.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(arg155_1.data_ptr()), c_void_p(arg299_1.data_ptr()), c_void_p(buf210.data_ptr()))
    del arg154_1
    del arg155_1
    del arg299_1
    del arg488_1
    del arg489_1
    # Source Nodes: [x_233, x_235, x_in_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf211 = extern_kernels.convolution(buf209, buf210, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf211, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf209
    buf212 = buf211; del buf211  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_75(c_void_p(buf212.data_ptr()), c_void_p(arg490_1.data_ptr()), c_void_p(arg491_1.data_ptr()), c_void_p(arg156_1.data_ptr()), c_void_p(arg157_1.data_ptr()))
    del arg156_1
    del arg157_1
    del arg490_1
    del arg491_1
    # Source Nodes: [x_236, x_238, x_in_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf213 = extern_kernels.convolution(buf212, arg300_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf213, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg300_1
    del buf212
    buf214 = empty_strided((8, 1984, 14, 14), (388864, 1, 27776, 1984), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_76(c_void_p(buf204.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(arg492_1.data_ptr()), c_void_p(arg493_1.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(arg159_1.data_ptr()), c_void_p(buf214.data_ptr()))
    del arg158_1
    del arg159_1
    del arg492_1
    del arg493_1
    # Source Nodes: [cat_90, x_239, x_241, x_in_101], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
    buf215 = extern_kernels.convolution(buf214, arg301_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf215, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg301_1
    del buf214
    buf216 = buf215; del buf215  # reuse
    buf217 = buf210; del buf210  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_77(c_void_p(buf216.data_ptr()), c_void_p(arg494_1.data_ptr()), c_void_p(arg495_1.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(arg161_1.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(buf217.data_ptr()))
    del arg160_1
    del arg161_1
    del arg302_1
    del arg494_1
    del arg495_1
    # Source Nodes: [x_242, x_244, x_in_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf218 = extern_kernels.convolution(buf216, buf217, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf218, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf216
    buf219 = buf218; del buf218  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_78(c_void_p(buf219.data_ptr()), c_void_p(arg496_1.data_ptr()), c_void_p(arg497_1.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(arg163_1.data_ptr()))
    del arg162_1
    del arg163_1
    del arg496_1
    del arg497_1
    # Source Nodes: [x_245, x_247, x_in_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf220 = extern_kernels.convolution(buf219, arg303_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf220, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg303_1
    del buf219
    buf221 = reinterpret_tensor(buf7, (8, 2048, 14, 14), (401408, 1, 28672, 2048), 0); del buf7  # reuse
    buf222 = buf221; del buf221  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_79(c_void_p(buf222.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(arg498_1.data_ptr()), c_void_p(arg499_1.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(arg165_1.data_ptr()))
    del arg164_1
    del arg165_1
    del arg498_1
    del arg499_1
    # Source Nodes: [x_248, x_250, x_in_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf223 = extern_kernels.convolution(buf222, arg304_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf223, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg304_1
    del buf222
    buf224 = buf223; del buf223  # reuse
    buf225 = buf217; del buf217  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_80(c_void_p(buf224.data_ptr()), c_void_p(arg500_1.data_ptr()), c_void_p(arg501_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(arg167_1.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(buf225.data_ptr()))
    del arg166_1
    del arg167_1
    del arg305_1
    del arg500_1
    del arg501_1
    # Source Nodes: [x_251, x_253, x_in_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf226 = extern_kernels.convolution(buf224, buf225, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf226, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf224
    buf227 = buf226; del buf226  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_81(c_void_p(buf227.data_ptr()), c_void_p(arg502_1.data_ptr()), c_void_p(arg503_1.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(arg169_1.data_ptr()))
    del arg168_1
    del arg169_1
    del arg502_1
    del arg503_1
    # Source Nodes: [x_254, x_256, x_in_107], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf228 = extern_kernels.convolution(buf227, arg306_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf228, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg306_1
    del buf227
    buf229 = empty_strided((8, 2112, 14, 14), (413952, 1, 29568, 2112), device='cpu', dtype=torch.float32)
    buf230 = buf229; del buf229  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_82(c_void_p(buf230.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(arg504_1.data_ptr()), c_void_p(arg505_1.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(arg171_1.data_ptr()))
    del arg170_1
    del arg171_1
    del arg504_1
    del arg505_1
    # Source Nodes: [x_257, x_259, x_in_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf231 = extern_kernels.convolution(buf230, arg307_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf231, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg307_1
    del buf230
    buf232 = buf231; del buf231  # reuse
    buf233 = buf225; del buf225  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_83(c_void_p(buf232.data_ptr()), c_void_p(arg506_1.data_ptr()), c_void_p(arg507_1.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(buf233.data_ptr()))
    del arg172_1
    del arg173_1
    del arg308_1
    del arg506_1
    del arg507_1
    # Source Nodes: [x_260, x_262, x_in_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf234 = extern_kernels.convolution(buf232, buf233, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf234, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf232
    buf235 = buf234; del buf234  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_84(c_void_p(buf235.data_ptr()), c_void_p(arg508_1.data_ptr()), c_void_p(arg509_1.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(arg175_1.data_ptr()))
    del arg174_1
    del arg175_1
    del arg508_1
    del arg509_1
    # Source Nodes: [x_263, x_265, x_in_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf236 = extern_kernels.convolution(buf235, arg309_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf236, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg309_1
    del buf235
    buf239 = empty((8, 2176, 14, 14), device='cpu', dtype=torch.float32)
    buf237 = reinterpret_tensor(buf239, (8, 1024, 14, 14), (426496, 196, 14, 1), 0)  # alias
    buf238 = reinterpret_tensor(buf239, (8, 1152, 14, 14), (426496, 196, 14, 1), 200704)  # alias
    buf240 = empty_strided((8, 2176, 14, 14), (426496, 1, 30464, 2176), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_cat_relu_85(c_void_p(buf204.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(arg510_1.data_ptr()), c_void_p(arg511_1.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf240.data_ptr()))
    del arg176_1
    del arg177_1
    del arg510_1
    del arg511_1
    del buf204
    del buf205
    del buf206
    del buf213
    del buf220
    del buf228
    del buf236
    # Source Nodes: [x_266, x_268, x_in_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf241 = extern_kernels.convolution(buf240, arg310_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf241, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg310_1
    del buf240
    buf242 = buf241; del buf241  # reuse
    buf243 = buf233; del buf233  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_86(c_void_p(buf242.data_ptr()), c_void_p(arg512_1.data_ptr()), c_void_p(arg513_1.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(buf243.data_ptr()))
    del arg178_1
    del arg179_1
    del arg311_1
    del arg512_1
    del arg513_1
    # Source Nodes: [x_269, x_271, x_in_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf244 = extern_kernels.convolution(buf242, buf243, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf244, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf242
    buf245 = buf244; del buf244  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_87(c_void_p(buf245.data_ptr()), c_void_p(arg514_1.data_ptr()), c_void_p(arg515_1.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(arg181_1.data_ptr()))
    del arg180_1
    del arg181_1
    del arg514_1
    del arg515_1
    # Source Nodes: [x_272, x_274, x_in_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf246 = extern_kernels.convolution(buf245, arg312_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf246, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg312_1
    del buf245
    buf247 = empty_strided((8, 2240, 14, 14), (439040, 1, 31360, 2240), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_88(c_void_p(buf237.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(arg516_1.data_ptr()), c_void_p(arg517_1.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(buf247.data_ptr()))
    del arg182_1
    del arg183_1
    del arg516_1
    del arg517_1
    # Source Nodes: [cat_82, x_275, x_277, x_in_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
    buf248 = extern_kernels.convolution(buf247, arg313_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf248, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg313_1
    del buf247
    buf249 = buf248; del buf248  # reuse
    buf250 = buf243; del buf243  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_89(c_void_p(buf249.data_ptr()), c_void_p(arg518_1.data_ptr()), c_void_p(arg519_1.data_ptr()), c_void_p(arg184_1.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(buf250.data_ptr()))
    del arg184_1
    del arg185_1
    del arg314_1
    del arg518_1
    del arg519_1
    # Source Nodes: [x_278, x_280, x_in_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf251 = extern_kernels.convolution(buf249, buf250, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf251, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf249
    buf252 = buf251; del buf251  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_90(c_void_p(buf252.data_ptr()), c_void_p(arg520_1.data_ptr()), c_void_p(arg521_1.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(arg187_1.data_ptr()))
    del arg186_1
    del arg187_1
    del arg520_1
    del arg521_1
    # Source Nodes: [x_281, x_283, x_in_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf253 = extern_kernels.convolution(buf252, arg315_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf253, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg315_1
    del buf252
    buf254 = reinterpret_tensor(buf95, (8, 2304, 14, 14), (451584, 1, 32256, 2304), 0); del buf95  # reuse
    buf255 = buf254; del buf254  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_91(c_void_p(buf255.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(arg522_1.data_ptr()), c_void_p(arg523_1.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(arg189_1.data_ptr()))
    del arg188_1
    del arg189_1
    del arg522_1
    del arg523_1
    # Source Nodes: [x_284, x_286, x_in_121], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf256 = extern_kernels.convolution(buf255, arg316_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf256, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg316_1
    del buf255
    buf257 = buf256; del buf256  # reuse
    buf258 = buf250; del buf250  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_92(c_void_p(buf257.data_ptr()), c_void_p(arg524_1.data_ptr()), c_void_p(arg525_1.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(buf258.data_ptr()))
    del arg190_1
    del arg191_1
    del arg317_1
    del arg524_1
    del arg525_1
    # Source Nodes: [x_287, x_289, x_in_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf259 = extern_kernels.convolution(buf257, buf258, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf259, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf257
    buf260 = buf259; del buf259  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_93(c_void_p(buf260.data_ptr()), c_void_p(arg526_1.data_ptr()), c_void_p(arg527_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(arg193_1.data_ptr()))
    del arg192_1
    del arg193_1
    del arg526_1
    del arg527_1
    # Source Nodes: [x_290, x_292, x_in_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf261 = extern_kernels.convolution(buf260, arg318_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf261, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg318_1
    del buf260
    buf262 = empty_strided((8, 2368, 14, 14), (464128, 1, 33152, 2368), device='cpu', dtype=torch.float32)
    buf263 = buf262; del buf262  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_94(c_void_p(buf263.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(arg528_1.data_ptr()), c_void_p(arg529_1.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(arg195_1.data_ptr()))
    del arg194_1
    del arg195_1
    del arg528_1
    del arg529_1
    # Source Nodes: [x_293, x_295, x_in_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf264 = extern_kernels.convolution(buf263, arg319_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf264, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del arg319_1
    del buf263
    buf265 = buf264; del buf264  # reuse
    buf266 = buf258; del buf258  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_95(c_void_p(buf265.data_ptr()), c_void_p(arg530_1.data_ptr()), c_void_p(arg531_1.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(buf266.data_ptr()))
    del arg196_1
    del arg197_1
    del arg320_1
    del arg530_1
    del arg531_1
    # Source Nodes: [x_296, x_298, x_in_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf267 = extern_kernels.convolution(buf265, buf266, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf267, (8, 800, 14, 14), (156800, 1, 11200, 800))
    del buf265
    del buf266
    buf268 = buf267; del buf267  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_96(c_void_p(buf268.data_ptr()), c_void_p(arg532_1.data_ptr()), c_void_p(arg533_1.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(arg199_1.data_ptr()))
    del arg198_1
    del arg199_1
    del arg532_1
    del arg533_1
    # Source Nodes: [x_299, x_301, x_in_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf269 = extern_kernels.convolution(buf268, arg321_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf269, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    del arg321_1
    del buf268
    buf270 = reinterpret_tensor(buf140, (8, 1408, 14, 14), (275968, 1, 19712, 1408), 0); del buf140  # reuse
    buf271 = empty_strided((8, 2432, 14, 14), (476672, 1, 34048, 2432), device='cpu', dtype=torch.float32)
    buf272 = empty_strided((8, 2432, 14, 14), (476672, 1, 34048, 2432), device='cpu', dtype=torch.float32)
    buf274 = empty_strided((8, 2432, 14, 14), (476672, 1, 34048, 2432), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_97(c_void_p(buf238.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(arg534_1.data_ptr()), c_void_p(arg535_1.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg536_1.data_ptr()), c_void_p(arg537_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg203_1.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf274.data_ptr()))
    del arg200_1
    del arg201_1
    del arg202_1
    del arg203_1
    del arg534_1
    del arg535_1
    del arg536_1
    del arg537_1
    del buf237
    del buf238
    del buf239
    del buf246
    del buf253
    del buf261
    del buf269
    del buf270
    del buf271
    # Source Nodes: [x_302, x_304, x_s_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf273 = extern_kernels.convolution(buf272, arg322_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf273, (8, 2304, 7, 7), (112896, 1, 16128, 2304))
    del arg322_1
    del buf272
    # Source Nodes: [x_305, x_307, x_in_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf275 = extern_kernels.convolution(buf274, arg323_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf275, (8, 1600, 14, 14), (313600, 1, 22400, 1600))
    del arg323_1
    del buf274
    buf276 = buf275; del buf275  # reuse
    buf277 = empty_strided((1600, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_98(c_void_p(buf276.data_ptr()), c_void_p(arg538_1.data_ptr()), c_void_p(arg539_1.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(buf277.data_ptr()))
    del arg204_1
    del arg205_1
    del arg324_1
    del arg538_1
    del arg539_1
    # Source Nodes: [x_308, x_310, x_in_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf278 = extern_kernels.convolution(buf276, buf277, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf278, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
    del buf276
    buf279 = buf278; del buf278  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_99(c_void_p(buf279.data_ptr()), c_void_p(arg540_1.data_ptr()), c_void_p(arg541_1.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(arg207_1.data_ptr()))
    del arg206_1
    del arg207_1
    del arg540_1
    del arg541_1
    # Source Nodes: [x_311, x_313, x_in_131], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf280 = extern_kernels.convolution(buf279, arg325_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf280, (8, 2176, 7, 7), (106624, 1, 15232, 2176))
    del arg325_1
    del buf279
    buf281 = empty_strided((8, 2432, 7, 7), (119168, 1, 17024, 2432), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_100(c_void_p(buf273.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(arg542_1.data_ptr()), c_void_p(arg543_1.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(arg209_1.data_ptr()), c_void_p(buf281.data_ptr()))
    del arg208_1
    del arg209_1
    del arg542_1
    del arg543_1
    # Source Nodes: [cat_74, x_314, x_316, x_in_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
    buf282 = extern_kernels.convolution(buf281, arg326_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf282, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
    del arg326_1
    del buf281
    buf283 = buf282; del buf282  # reuse
    buf284 = buf277; del buf277  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_101(c_void_p(buf283.data_ptr()), c_void_p(arg544_1.data_ptr()), c_void_p(arg545_1.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg327_1.data_ptr()), c_void_p(buf284.data_ptr()))
    del arg210_1
    del arg211_1
    del arg327_1
    del arg544_1
    del arg545_1
    # Source Nodes: [x_317, x_319, x_in_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf285 = extern_kernels.convolution(buf283, buf284, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf285, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
    del buf283
    buf286 = buf285; del buf285  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_102(c_void_p(buf286.data_ptr()), c_void_p(arg546_1.data_ptr()), c_void_p(arg547_1.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(arg213_1.data_ptr()))
    del arg212_1
    del arg213_1
    del arg546_1
    del arg547_1
    # Source Nodes: [x_320, x_322, x_in_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf287 = extern_kernels.convolution(buf286, arg328_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf287, (8, 2176, 7, 7), (106624, 1, 15232, 2176))
    del arg328_1
    del buf286
    buf288 = empty_strided((8, 2560, 7, 7), (125440, 1, 17920, 2560), device='cpu', dtype=torch.float32)
    buf289 = buf288; del buf288  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_103(c_void_p(buf289.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(arg548_1.data_ptr()), c_void_p(arg549_1.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(arg215_1.data_ptr()))
    del arg214_1
    del arg215_1
    del arg548_1
    del arg549_1
    # Source Nodes: [x_323, x_325, x_in_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf290 = extern_kernels.convolution(buf289, arg329_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf290, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
    del arg329_1
    del buf289
    buf291 = buf290; del buf290  # reuse
    buf292 = buf284; del buf284  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_104(c_void_p(buf291.data_ptr()), c_void_p(arg550_1.data_ptr()), c_void_p(arg551_1.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg330_1.data_ptr()), c_void_p(buf292.data_ptr()))
    del arg216_1
    del arg217_1
    del arg330_1
    del arg550_1
    del arg551_1
    # Source Nodes: [x_326, x_328, x_in_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf293 = extern_kernels.convolution(buf291, buf292, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
    assert_size_stride(buf293, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
    del buf291
    del buf292
    buf294 = buf293; del buf293  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_105(c_void_p(buf294.data_ptr()), c_void_p(arg552_1.data_ptr()), c_void_p(arg553_1.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(arg219_1.data_ptr()))
    del arg218_1
    del arg219_1
    del arg552_1
    del arg553_1
    # Source Nodes: [x_329, x_331, x_in_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf295 = extern_kernels.convolution(buf294, arg331_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf295, (8, 2176, 7, 7), (106624, 1, 15232, 2176))
    del arg331_1
    del buf294
    buf296 = empty_strided((8, 2688, 7, 7), (131712, 1, 18816, 2688), device='cpu', dtype=torch.float32)
    buf297 = empty_strided((8, 2688, 1, 1), (2688, 1, 21504, 21504), device='cpu', dtype=torch.float32)
    buf298 = reinterpret_tensor(buf297, (8, 2688, 1, 1), (2688, 1, 2688, 2688), 0); del buf297  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_relu_106(c_void_p(buf298.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(arg554_1.data_ptr()), c_void_p(arg555_1.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(buf296.data_ptr()))
    del arg220_1
    del arg221_1
    del arg554_1
    del arg555_1
    del buf273
    del buf280
    del buf287
    del buf295
    del buf296
    # Source Nodes: [x_333, x_336, x_337, x_340], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
    buf299 = extern_kernels.convolution(buf298, arg332_1, arg333_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf299, (8, 1000, 1, 1), (1000, 1, 1000, 1000))
    del arg332_1
    del arg333_1
    return (reinterpret_tensor(buf299, (8, 1000), (1000, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((316, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((316, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((356, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((356, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((376, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((376, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((376, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((376, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((1088, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((1088, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((1216, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((1216, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((1344, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((1344, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((1408, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((1408, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((1472, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((1472, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((1664, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((1664, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((1728, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((1728, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((1792, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((1792, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((1856, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((1856, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((1984, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((1984, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((2112, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((2112, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((2176, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((2176, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((2368, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((2368, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((2432, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((2432, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((2432, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((2432, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((2432, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((2432, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((2688, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((2688, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((128, 3, 7, 7), (147, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((296, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((200, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((200, 316, 1, 1), (316, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((200, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((200, 356, 1, 1), (356, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((640, 376, 1, 1), (376, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((400, 376, 1, 1), (376, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((400, 704, 1, 1), (704, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((400, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((400, 832, 1, 1), (832, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((400, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((400, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((400, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((400, 1088, 1, 1), (1088, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((1152, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((800, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((800, 1216, 1, 1), (1216, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((800, 1280, 1, 1), (1280, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((800, 1344, 1, 1), (1344, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((800, 1408, 1, 1), (1408, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((800, 1472, 1, 1), (1472, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((800, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((800, 1600, 1, 1), (1600, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((800, 1664, 1, 1), (1664, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((800, 1728, 1, 1), (1728, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((800, 1792, 1, 1), (1792, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((800, 1856, 1, 1), (1856, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((800, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((800, 1984, 1, 1), (1984, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((800, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((800, 2112, 1, 1), (2112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((800, 2176, 1, 1), (2176, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((800, 2240, 1, 1), (2240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((800, 2304, 1, 1), (2304, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((800, 2368, 1, 1), (2368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((2304, 2432, 1, 1), (2432, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((1600, 2432, 1, 1), (2432, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((1600, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((2176, 1600, 1, 1), (1600, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((1600, 2432, 1, 1), (2432, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((1600, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((2176, 1600, 1, 1), (1600, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((1600, 2560, 1, 1), (2560, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((1600, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((2176, 1600, 1, 1), (1600, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((1000, 2688, 1, 1), (2688, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((316, ), (1, ), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((316, ), (1, ), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((356, ), (1, ), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((356, ), (1, ), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((376, ), (1, ), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((376, ), (1, ), device='cpu', dtype=torch.float32)
    arg364_1 = rand_strided((376, ), (1, ), device='cpu', dtype=torch.float32)
    arg365_1 = rand_strided((376, ), (1, ), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg367_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg368_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg369_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg370_1 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    arg371_1 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    arg372_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg373_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg374_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg375_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg376_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg377_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg378_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg379_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg380_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg381_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg382_1 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    arg383_1 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    arg384_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg385_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg386_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg387_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg388_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg389_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg390_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg391_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg392_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg393_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg394_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg395_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg396_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg397_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg398_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg399_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg400_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg401_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg402_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg403_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg404_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg405_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg406_1 = rand_strided((1088, ), (1, ), device='cpu', dtype=torch.float32)
    arg407_1 = rand_strided((1088, ), (1, ), device='cpu', dtype=torch.float32)
    arg408_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg409_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg410_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg411_1 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    arg412_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg413_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg414_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg415_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg416_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg417_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg418_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg419_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg420_1 = rand_strided((1216, ), (1, ), device='cpu', dtype=torch.float32)
    arg421_1 = rand_strided((1216, ), (1, ), device='cpu', dtype=torch.float32)
    arg422_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg423_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg424_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg425_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg426_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg427_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg428_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg429_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg430_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg431_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg432_1 = rand_strided((1344, ), (1, ), device='cpu', dtype=torch.float32)
    arg433_1 = rand_strided((1344, ), (1, ), device='cpu', dtype=torch.float32)
    arg434_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg435_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg436_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg437_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg438_1 = rand_strided((1408, ), (1, ), device='cpu', dtype=torch.float32)
    arg439_1 = rand_strided((1408, ), (1, ), device='cpu', dtype=torch.float32)
    arg440_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg441_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg442_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg443_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg444_1 = rand_strided((1472, ), (1, ), device='cpu', dtype=torch.float32)
    arg445_1 = rand_strided((1472, ), (1, ), device='cpu', dtype=torch.float32)
    arg446_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg447_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg448_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg449_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg450_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg451_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg452_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg453_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg454_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg455_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg456_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg457_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg458_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg459_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg460_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg461_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg462_1 = rand_strided((1664, ), (1, ), device='cpu', dtype=torch.float32)
    arg463_1 = rand_strided((1664, ), (1, ), device='cpu', dtype=torch.float32)
    arg464_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg465_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg466_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg467_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg468_1 = rand_strided((1728, ), (1, ), device='cpu', dtype=torch.float32)
    arg469_1 = rand_strided((1728, ), (1, ), device='cpu', dtype=torch.float32)
    arg470_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg471_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg472_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg473_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg474_1 = rand_strided((1792, ), (1, ), device='cpu', dtype=torch.float32)
    arg475_1 = rand_strided((1792, ), (1, ), device='cpu', dtype=torch.float32)
    arg476_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg477_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg478_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg479_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg480_1 = rand_strided((1856, ), (1, ), device='cpu', dtype=torch.float32)
    arg481_1 = rand_strided((1856, ), (1, ), device='cpu', dtype=torch.float32)
    arg482_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg483_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg484_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg485_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg486_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg487_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg488_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg489_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg490_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg491_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg492_1 = rand_strided((1984, ), (1, ), device='cpu', dtype=torch.float32)
    arg493_1 = rand_strided((1984, ), (1, ), device='cpu', dtype=torch.float32)
    arg494_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg495_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg496_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg497_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg498_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg499_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg500_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg501_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg502_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg503_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg504_1 = rand_strided((2112, ), (1, ), device='cpu', dtype=torch.float32)
    arg505_1 = rand_strided((2112, ), (1, ), device='cpu', dtype=torch.float32)
    arg506_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg507_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg508_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg509_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg510_1 = rand_strided((2176, ), (1, ), device='cpu', dtype=torch.float32)
    arg511_1 = rand_strided((2176, ), (1, ), device='cpu', dtype=torch.float32)
    arg512_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg513_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg514_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg515_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg516_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg517_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg518_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg519_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg520_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg521_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg522_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg523_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg524_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg525_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg526_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg527_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg528_1 = rand_strided((2368, ), (1, ), device='cpu', dtype=torch.float32)
    arg529_1 = rand_strided((2368, ), (1, ), device='cpu', dtype=torch.float32)
    arg530_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg531_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg532_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg533_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg534_1 = rand_strided((2432, ), (1, ), device='cpu', dtype=torch.float32)
    arg535_1 = rand_strided((2432, ), (1, ), device='cpu', dtype=torch.float32)
    arg536_1 = rand_strided((2432, ), (1, ), device='cpu', dtype=torch.float32)
    arg537_1 = rand_strided((2432, ), (1, ), device='cpu', dtype=torch.float32)
    arg538_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg539_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg540_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg541_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg542_1 = rand_strided((2432, ), (1, ), device='cpu', dtype=torch.float32)
    arg543_1 = rand_strided((2432, ), (1, ), device='cpu', dtype=torch.float32)
    arg544_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg545_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg546_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg547_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg548_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg549_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg550_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg551_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg552_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg553_1 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    arg554_1 = rand_strided((2688, ), (1, ), device='cpu', dtype=torch.float32)
    arg555_1 = rand_strided((2688, ), (1, ), device='cpu', dtype=torch.float32)
    arg556_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dpn107', benchmark_compiled_module)
