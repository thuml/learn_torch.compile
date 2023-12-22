
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(65536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (65536L*x1) + (196608L*x0))];
                        out_ptr0[static_cast<long>(x1 + (3L*x2) + (196608L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x2 + (9L*x1) + (27L*x0))];
                            out_ptr1[static_cast<long>(x1 + (3L*x2) + (27L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_silu_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (216L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (24L*x2) + (216L*x0)), static_cast<long>(24L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (216L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (24L*x2) + (216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_silu_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_silu_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-1L) + (2L*x1));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(128);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<int>((-1L) + (2L*x2));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_out_ptr0 + static_cast<long>((-8256L) + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0)), to_float_mask(tmp10));
                                auto tmp13 = decltype(tmp12)(1)/(decltype(tmp12)(1) + tmp12.neg().exp());
                                auto tmp14 = tmp12 * tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp11(), to_float_mask(tmp10));
                            auto tmp16 = c10::convert<int>(2L*x2);
                            auto tmp17 = tmp16 >= tmp1;
                            auto tmp18 = tmp16 < tmp3;
                            auto tmp19 = tmp17 & tmp18;
                            auto tmp20 = tmp5 & tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_out_ptr0 + static_cast<long>((-8192L) + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0)), to_float_mask(tmp20));
                                auto tmp23 = decltype(tmp22)(1)/(decltype(tmp22)(1) + tmp22.neg().exp());
                                auto tmp24 = tmp22 * tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp21(), to_float_mask(tmp20));
                            auto tmp26 = at::vec::maximum(tmp25, tmp15);
                            auto tmp27 = c10::convert<int>(1L + (2L*x2));
                            auto tmp28 = tmp27 >= tmp1;
                            auto tmp29 = tmp27 < tmp3;
                            auto tmp30 = tmp28 & tmp29;
                            auto tmp31 = tmp5 & tmp30;
                            auto tmp32 = [&]
                            {
                                auto tmp33 = masked_load(in_out_ptr0 + static_cast<long>((-8128L) + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0)), to_float_mask(tmp31));
                                auto tmp34 = decltype(tmp33)(1)/(decltype(tmp33)(1) + tmp33.neg().exp());
                                auto tmp35 = tmp33 * tmp34;
                                return tmp35;
                            }
                            ;
                            auto tmp36 = decltype(tmp32())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp32(), to_float_mask(tmp31));
                            auto tmp37 = at::vec::maximum(tmp36, tmp26);
                            auto tmp38 = c10::convert<int>(2L*x1);
                            auto tmp39 = tmp38 >= tmp1;
                            auto tmp40 = tmp38 < tmp3;
                            auto tmp41 = tmp39 & tmp40;
                            auto tmp42 = tmp41 & tmp9;
                            auto tmp43 = [&]
                            {
                                auto tmp44 = masked_load(in_out_ptr0 + static_cast<long>((-64L) + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0)), to_float_mask(tmp42));
                                auto tmp45 = decltype(tmp44)(1)/(decltype(tmp44)(1) + tmp44.neg().exp());
                                auto tmp46 = tmp44 * tmp45;
                                return tmp46;
                            }
                            ;
                            auto tmp47 = decltype(tmp43())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp43(), to_float_mask(tmp42));
                            auto tmp48 = at::vec::maximum(tmp47, tmp37);
                            auto tmp49 = tmp41 & tmp19;
                            auto tmp50 = [&]
                            {
                                auto tmp51 = masked_load(in_out_ptr0 + static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (1048576L*x0)), to_float_mask(tmp49));
                                auto tmp52 = decltype(tmp51)(1)/(decltype(tmp51)(1) + tmp51.neg().exp());
                                auto tmp53 = tmp51 * tmp52;
                                return tmp53;
                            }
                            ;
                            auto tmp54 = decltype(tmp50())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp50(), to_float_mask(tmp49));
                            auto tmp55 = at::vec::maximum(tmp54, tmp48);
                            auto tmp56 = tmp41 & tmp30;
                            auto tmp57 = [&]
                            {
                                auto tmp58 = masked_load(in_out_ptr0 + static_cast<long>(64L + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0)), to_float_mask(tmp56));
                                auto tmp59 = decltype(tmp58)(1)/(decltype(tmp58)(1) + tmp58.neg().exp());
                                auto tmp60 = tmp58 * tmp59;
                                return tmp60;
                            }
                            ;
                            auto tmp61 = decltype(tmp57())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp57(), to_float_mask(tmp56));
                            auto tmp62 = at::vec::maximum(tmp61, tmp55);
                            auto tmp63 = c10::convert<int>(1L + (2L*x1));
                            auto tmp64 = tmp63 >= tmp1;
                            auto tmp65 = tmp63 < tmp3;
                            auto tmp66 = tmp64 & tmp65;
                            auto tmp67 = tmp66 & tmp9;
                            auto tmp68 = [&]
                            {
                                auto tmp69 = masked_load(in_out_ptr0 + static_cast<long>(8128L + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0)), to_float_mask(tmp67));
                                auto tmp70 = decltype(tmp69)(1)/(decltype(tmp69)(1) + tmp69.neg().exp());
                                auto tmp71 = tmp69 * tmp70;
                                return tmp71;
                            }
                            ;
                            auto tmp72 = decltype(tmp68())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp68(), to_float_mask(tmp67));
                            auto tmp73 = at::vec::maximum(tmp72, tmp62);
                            auto tmp74 = tmp66 & tmp19;
                            auto tmp75 = [&]
                            {
                                auto tmp76 = masked_load(in_out_ptr0 + static_cast<long>(8192L + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0)), to_float_mask(tmp74));
                                auto tmp77 = decltype(tmp76)(1)/(decltype(tmp76)(1) + tmp76.neg().exp());
                                auto tmp78 = tmp76 * tmp77;
                                return tmp78;
                            }
                            ;
                            auto tmp79 = decltype(tmp75())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp75(), to_float_mask(tmp74));
                            auto tmp80 = at::vec::maximum(tmp79, tmp73);
                            auto tmp81 = tmp66 & tmp30;
                            auto tmp82 = [&]
                            {
                                auto tmp83 = masked_load(in_out_ptr0 + static_cast<long>(8256L + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0)), to_float_mask(tmp81));
                                auto tmp84 = decltype(tmp83)(1)/(decltype(tmp83)(1) + tmp83.neg().exp());
                                auto tmp85 = tmp83 * tmp84;
                                return tmp85;
                            }
                            ;
                            auto tmp86 = decltype(tmp82())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp82(), to_float_mask(tmp81));
                            auto tmp87 = at::vec::maximum(tmp86, tmp80);
                            tmp87.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (4096L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_silu_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_mean_silu_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(4096L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x2) + (262144L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_silu_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_silu_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_mean_silu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(4096L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x2) + (262144L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_silu_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_silu_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
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


cpp_fused__native_batch_norm_legit_no_training_mean_silu_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x2) + (131072L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_silu_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_silu_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
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


cpp_fused__native_batch_norm_legit_no_training_mean_silu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x2) + (131072L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_silu_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_silu_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_mean_silu_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x2) + (65536L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_silu_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(((8L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>(x2) % static_cast<long>(8L)))) % static_cast<long>(16L))) + (2048L*(static_cast<long>(c10::div_floor_integer(((8L*(static_cast<long>(x1) % static_cast<long>(2L))) + (16L*(c10::div_floor_integer(x2, 8L))) + (128L*(c10::div_floor_integer(x1, 2L))) + (static_cast<long>(x2) % static_cast<long>(8L))), 16L)) % static_cast<long>(16L))) + (32768L*(static_cast<long>(c10::div_floor_integer(((8L*(static_cast<long>(x1) % static_cast<long>(2L))) + (16L*(c10::div_floor_integer(x2, 8L))) + (128L*(c10::div_floor_integer(x1, 2L))) + (256L*x3) + (256L*x3_inner) + (4096L*x0) + (static_cast<long>(x2) % static_cast<long>(8L))), 32768L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((8L*(static_cast<long>(x1) % static_cast<long>(2L))) + (16L*(c10::div_floor_integer(x2, 8L))) + (128L*(c10::div_floor_integer(x1, 2L))) + (256L*x3) + (256L*x3_inner) + (4096L*x0) + (static_cast<long>(x2) % static_cast<long>(8L))), 256L)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (1024L*x1) + (4096L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (16L*x2) + (1024L*x1) + (4096L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(144L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>((-2L) + (8L*(c10::div_floor_integer((x3 + (144L*x1)), 288L))) + (c10::div_floor_integer(x3, 12L)));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(16);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<long>((-2L) + (8L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>(x3) % static_cast<long>(12L)));
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-13056L) + (384L*(static_cast<long>(x3) % static_cast<long>(12L))) + (3072L*(static_cast<long>(x1) % static_cast<long>(2L))) + (6144L*(c10::div_floor_integer(x3, 12L))) + (49152L*(c10::div_floor_integer((x3 + (144L*x1)), 288L))) + (98304L*(c10::div_floor_integer((x3 + (144L*x1) + (576L*x2) + (27648L*x0)), 221184L))) + (static_cast<long>(c10::div_floor_integer((x3 + (144L*x1) + (576L*x2) + (27648L*x0)), 576L)) % static_cast<long>(384L)))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            out_ptr2[static_cast<long>(x3 + (144L*x2) + (2304L*x1) + (9216L*x0))] = tmp13;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>((x1 + (8L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))))) % static_cast<long>(16L))) + (2048L*(static_cast<long>(c10::div_floor_integer((x1 + (8L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))) + (16L*x2) + (128L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L)))), 16L)) % static_cast<long>(16L))) + (32768L*(static_cast<long>(c10::div_floor_integer((x1 + (8L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))) + (16L*x2) + (128L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (256L*x3) + (256L*x3_inner) + (4096L*(c10::div_floor_integer(x0, 4L)))), 32768L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer((x1 + (8L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))) + (16L*x2) + (128L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (256L*x3) + (256L*x3_inner) + (4096L*(c10::div_floor_integer(x0, 4L)))), 256L)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (128L*x1) + (1024L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_mul_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (144L*x1) + (9216L*x0))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp3 = c10::convert<long>(11L + (23L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 12L)));
                            auto tmp4 = static_cast<long>(192);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(static_cast<long>((11L + (23L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 12L)))) % static_cast<long>(24L));
                                auto tmp8 = static_cast<long>(23);
                                auto tmp9 = tmp7 < tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_ptr1[static_cast<long>((23L*(c10::div_floor_integer((11L + (23L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 12L))), 24L))) + (184L*(static_cast<long>(x1) % static_cast<long>(8L))) + (1472L*x0) + (static_cast<long>((11L + (23L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 12L)))) % static_cast<long>(24L)))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp14 = c10::convert<long>(11L + (23L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(12L)));
                            auto tmp15 = tmp14 < tmp4;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(static_cast<long>((11L + (23L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(12L)))) % static_cast<long>(24L));
                                auto tmp18 = static_cast<long>(23);
                                auto tmp19 = tmp17 < tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = in_ptr2[static_cast<long>((23L*(static_cast<long>(c10::div_floor_integer((11L + (23L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(12L))), 24L)) % static_cast<long>(8L))) + (184L*(c10::div_floor_integer(x1, 8L))) + (1472L*x0) + (static_cast<long>((11L + (23L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(12L)))) % static_cast<long>(24L)))];
                                    return tmp21;
                                }
                                ;
                                auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp24 = decltype(tmp13)(tmp13 + tmp23);
                            auto tmp25 = decltype(tmp2)(tmp2 + tmp24);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp25);
                        }
                        out_ptr0[static_cast<long>(x1 + (64L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (144L*x1) + (9216L*x0))];
                        auto tmp26 = out_ptr0[static_cast<long>(x1 + (64L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = c10::convert<long>(11L + (23L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 12L)));
                        auto tmp4 = static_cast<long>(192);
                        auto tmp5 = tmp3 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = c10::convert<long>(static_cast<long>((11L + (23L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 12L)))) % static_cast<long>(24L));
                            auto tmp8 = static_cast<long>(23);
                            auto tmp9 = tmp7 < tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = in_ptr1[static_cast<long>((23L*(c10::div_floor_integer((11L + (23L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 12L))), 24L))) + (184L*(static_cast<long>(x1) % static_cast<long>(8L))) + (1472L*x0) + (static_cast<long>((11L + (23L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 12L)))) % static_cast<long>(24L)))];
                                return tmp11;
                            }
                            ;
                            auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp14 = c10::convert<long>(11L + (23L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(12L)));
                        auto tmp15 = tmp14 < tmp4;
                        auto tmp16 = [&]
                        {
                            auto tmp17 = c10::convert<long>(static_cast<long>((11L + (23L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(12L)))) % static_cast<long>(24L));
                            auto tmp18 = static_cast<long>(23);
                            auto tmp19 = tmp17 < tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = in_ptr2[static_cast<long>((23L*(static_cast<long>(c10::div_floor_integer((11L + (23L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(12L))), 24L)) % static_cast<long>(8L))) + (184L*(c10::div_floor_integer(x1, 8L))) + (1472L*x0) + (static_cast<long>((11L + (23L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(12L)))) % static_cast<long>(24L)))];
                                return tmp21;
                            }
                            ;
                            auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                        auto tmp24 = decltype(tmp13)(tmp13 + tmp23);
                        auto tmp25 = decltype(tmp2)(tmp2 + tmp24);
                        auto tmp27 = decltype(tmp25)(tmp25 - tmp26);
                        auto tmp28 = std::exp(tmp27);
                        in_out_ptr0[static_cast<long>(x2 + (144L*x1) + (9216L*x0))] = tmp28;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-2L) + (8L*(c10::div_floor_integer((x2 + (144L*x1)), 288L))) + (c10::div_floor_integer(x2, 12L)));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(16);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<int>((-2L) + (8L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>(x2) % static_cast<long>(12L)));
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr3[static_cast<long>((-13056L) + (384L*(static_cast<long>(x2) % static_cast<long>(12L))) + (3072L*(static_cast<long>(x1) % static_cast<long>(2L))) + (6144L*(c10::div_floor_integer(x2, 12L))) + (49152L*(c10::div_floor_integer((x2 + (144L*x1)), 288L))) + (98304L*(c10::div_floor_integer((9216L + x2 + (144L*x1) + (576L*x3) + (576L*x3_inner) + (27648L*x0)), 221184L))) + (static_cast<long>(c10::div_floor_integer((9216L + x2 + (144L*x1) + (576L*x3) + (576L*x3_inner) + (27648L*x0)), 576L)) % static_cast<long>(384L)))]; return masked_load(tmpbuf, to_float_mask(tmp10)); })();
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            tmp13.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (4608L*x1) + (18432L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x2) % static_cast<long>(8L))) + (256L*(static_cast<long>(x1) % static_cast<long>(8L))) + (2048L*(c10::div_floor_integer(x2, 8L))) + (4096L*(c10::div_floor_integer(x1, 8L))) + (8192L*(c10::div_floor_integer((x3 + x3_inner), 32L))) + (65536L*x0) + (static_cast<long>((x3 + x3_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3));
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
                            tmp16.store(out_ptr0 + static_cast<long>(x3 + (256L*x2) + (4096L*x1) + (65536L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x2) + (65536L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x2) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_silu_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(((4L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>(x2) % static_cast<long>(4L)))) % static_cast<long>(8L))) + (1024L*(static_cast<long>(c10::div_floor_integer(((4L*(static_cast<long>(x1) % static_cast<long>(2L))) + (8L*(c10::div_floor_integer(x2, 4L))) + (32L*(c10::div_floor_integer(x1, 2L))) + (static_cast<long>(x2) % static_cast<long>(4L))), 8L)) % static_cast<long>(8L))) + (8192L*(static_cast<long>(c10::div_floor_integer(((4L*(static_cast<long>(x1) % static_cast<long>(2L))) + (8L*(c10::div_floor_integer(x2, 4L))) + (32L*(c10::div_floor_integer(x1, 2L))) + (64L*x3) + (64L*x3_inner) + (1024L*x0) + (static_cast<long>(x2) % static_cast<long>(4L))), 8192L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((4L*(static_cast<long>(x1) % static_cast<long>(2L))) + (8L*(c10::div_floor_integer(x2, 4L))) + (32L*(c10::div_floor_integer(x1, 2L))) + (64L*x3) + (64L*x3_inner) + (1024L*x0) + (static_cast<long>(x2) % static_cast<long>(4L))), 64L)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (1024L*x0)));
                        tmp0.store(out_ptr1 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (1024L*x0)));
                    }
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(144L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>((-2L) + (8L*(c10::div_floor_integer((x3 + (144L*x1)), 288L))) + (c10::div_floor_integer(x3, 12L)));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(16);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<long>((-2L) + (8L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>(x3) % static_cast<long>(12L)));
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-21760L) + (640L*(static_cast<long>(x3) % static_cast<long>(12L))) + (5120L*(static_cast<long>(x1) % static_cast<long>(2L))) + (10240L*(c10::div_floor_integer(x3, 12L))) + (81920L*(c10::div_floor_integer((x3 + (144L*x1)), 288L))) + (163840L*(c10::div_floor_integer((x3 + (144L*x1) + (576L*x2) + (46080L*x0)), 368640L))) + (static_cast<long>(c10::div_floor_integer((x3 + (144L*x1) + (576L*x2) + (46080L*x0)), 576L)) % static_cast<long>(640L)))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            out_ptr2[static_cast<long>(x3 + (144L*x2) + (2304L*x1) + (9216L*x0))] = tmp13;
                        }
                    }
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>((x1 + (4L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))))) % static_cast<long>(8L))) + (1024L*(static_cast<long>(c10::div_floor_integer((x1 + (4L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))) + (8L*x2) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L)))), 8L)) % static_cast<long>(8L))) + (8192L*(static_cast<long>(c10::div_floor_integer((x1 + (4L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))) + (8L*x2) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*x3) + (64L*x3_inner) + (1024L*(c10::div_floor_integer(x0, 4L)))), 8192L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer((x1 + (4L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))) + (8L*x2) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*x3) + (64L*x3_inner) + (1024L*(c10::div_floor_integer(x0, 4L)))), 64L)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (64L*x1) + (256L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_mul_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (144L*x1) + (2304L*x0))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp3 = c10::convert<long>(11L + (23L*(c10::div_floor_integer(x1, 4L))) + (c10::div_floor_integer(x2, 12L)));
                            auto tmp4 = static_cast<long>(96);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(static_cast<long>((11L + (23L*(c10::div_floor_integer(x1, 4L))) + (c10::div_floor_integer(x2, 12L)))) % static_cast<long>(24L));
                                auto tmp8 = static_cast<long>(23);
                                auto tmp9 = tmp7 < tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_ptr1[static_cast<long>((23L*(c10::div_floor_integer((11L + (23L*(c10::div_floor_integer(x1, 4L))) + (c10::div_floor_integer(x2, 12L))), 24L))) + (92L*(static_cast<long>(x1) % static_cast<long>(4L))) + (368L*x0) + (static_cast<long>((11L + (23L*(c10::div_floor_integer(x1, 4L))) + (c10::div_floor_integer(x2, 12L)))) % static_cast<long>(24L)))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp14 = c10::convert<long>(11L + (23L*(static_cast<long>(x1) % static_cast<long>(4L))) + (static_cast<long>(x2) % static_cast<long>(12L)));
                            auto tmp15 = tmp14 < tmp4;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(static_cast<long>((11L + (23L*(static_cast<long>(x1) % static_cast<long>(4L))) + (static_cast<long>(x2) % static_cast<long>(12L)))) % static_cast<long>(24L));
                                auto tmp18 = static_cast<long>(23);
                                auto tmp19 = tmp17 < tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = in_ptr2[static_cast<long>((23L*(static_cast<long>(c10::div_floor_integer((11L + (23L*(static_cast<long>(x1) % static_cast<long>(4L))) + (static_cast<long>(x2) % static_cast<long>(12L))), 24L)) % static_cast<long>(4L))) + (92L*(c10::div_floor_integer(x1, 4L))) + (368L*x0) + (static_cast<long>((11L + (23L*(static_cast<long>(x1) % static_cast<long>(4L))) + (static_cast<long>(x2) % static_cast<long>(12L)))) % static_cast<long>(24L)))];
                                    return tmp21;
                                }
                                ;
                                auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp24 = decltype(tmp13)(tmp13 + tmp23);
                            auto tmp25 = decltype(tmp2)(tmp2 + tmp24);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp25);
                        }
                        out_ptr0[static_cast<long>(x1 + (16L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (144L*x1) + (2304L*x0))];
                        auto tmp26 = out_ptr0[static_cast<long>(x1 + (16L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = c10::convert<long>(11L + (23L*(c10::div_floor_integer(x1, 4L))) + (c10::div_floor_integer(x2, 12L)));
                        auto tmp4 = static_cast<long>(96);
                        auto tmp5 = tmp3 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = c10::convert<long>(static_cast<long>((11L + (23L*(c10::div_floor_integer(x1, 4L))) + (c10::div_floor_integer(x2, 12L)))) % static_cast<long>(24L));
                            auto tmp8 = static_cast<long>(23);
                            auto tmp9 = tmp7 < tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = in_ptr1[static_cast<long>((23L*(c10::div_floor_integer((11L + (23L*(c10::div_floor_integer(x1, 4L))) + (c10::div_floor_integer(x2, 12L))), 24L))) + (92L*(static_cast<long>(x1) % static_cast<long>(4L))) + (368L*x0) + (static_cast<long>((11L + (23L*(c10::div_floor_integer(x1, 4L))) + (c10::div_floor_integer(x2, 12L)))) % static_cast<long>(24L)))];
                                return tmp11;
                            }
                            ;
                            auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp14 = c10::convert<long>(11L + (23L*(static_cast<long>(x1) % static_cast<long>(4L))) + (static_cast<long>(x2) % static_cast<long>(12L)));
                        auto tmp15 = tmp14 < tmp4;
                        auto tmp16 = [&]
                        {
                            auto tmp17 = c10::convert<long>(static_cast<long>((11L + (23L*(static_cast<long>(x1) % static_cast<long>(4L))) + (static_cast<long>(x2) % static_cast<long>(12L)))) % static_cast<long>(24L));
                            auto tmp18 = static_cast<long>(23);
                            auto tmp19 = tmp17 < tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = in_ptr2[static_cast<long>((23L*(static_cast<long>(c10::div_floor_integer((11L + (23L*(static_cast<long>(x1) % static_cast<long>(4L))) + (static_cast<long>(x2) % static_cast<long>(12L))), 24L)) % static_cast<long>(4L))) + (92L*(c10::div_floor_integer(x1, 4L))) + (368L*x0) + (static_cast<long>((11L + (23L*(static_cast<long>(x1) % static_cast<long>(4L))) + (static_cast<long>(x2) % static_cast<long>(12L)))) % static_cast<long>(24L)))];
                                return tmp21;
                            }
                            ;
                            auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                        auto tmp24 = decltype(tmp13)(tmp13 + tmp23);
                        auto tmp25 = decltype(tmp2)(tmp2 + tmp24);
                        auto tmp27 = decltype(tmp25)(tmp25 - tmp26);
                        auto tmp28 = std::exp(tmp27);
                        in_out_ptr0[static_cast<long>(x2 + (144L*x1) + (2304L*x0))] = tmp28;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-2L) + (8L*(c10::div_floor_integer((x2 + (144L*x1)), 288L))) + (c10::div_floor_integer(x2, 12L)));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(16);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<int>((-2L) + (8L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>(x2) % static_cast<long>(12L)));
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr3[static_cast<long>((-21760L) + (640L*(static_cast<long>(x2) % static_cast<long>(12L))) + (5120L*(static_cast<long>(x1) % static_cast<long>(2L))) + (10240L*(c10::div_floor_integer(x2, 12L))) + (81920L*(c10::div_floor_integer((x2 + (144L*x1)), 288L))) + (163840L*(c10::div_floor_integer((9216L + x2 + (144L*x1) + (576L*x3) + (576L*x3_inner) + (46080L*x0)), 368640L))) + (static_cast<long>(c10::div_floor_integer((9216L + x2 + (144L*x1) + (576L*x3) + (576L*x3_inner) + (46080L*x0)), 576L)) % static_cast<long>(640L)))]; return masked_load(tmpbuf, to_float_mask(tmp10)); })();
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            tmp13.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (9216L*x1) + (36864L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x2) % static_cast<long>(4L))) + (256L*(static_cast<long>(x1) % static_cast<long>(4L))) + (1024L*(c10::div_floor_integer(x2, 4L))) + (2048L*(c10::div_floor_integer(x1, 4L))) + (4096L*(c10::div_floor_integer((x3 + x3_inner), 64L))) + (32768L*x0) + (static_cast<long>((x3 + x3_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3));
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
                            tmp16.store(out_ptr0 + static_cast<long>(x3 + (512L*x2) + (4096L*x1) + (32768L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_silu_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bmm_constant_pad_nd_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x0) % static_cast<long>(8L))) + (1024L*(static_cast<long>(c10::div_floor_integer(((8L*(c10::div_floor_integer(x0, 8L))) + (static_cast<long>(x0) % static_cast<long>(8L))), 8L)) % static_cast<long>(8L))) + (8192L*(static_cast<long>(c10::div_floor_integer(((8L*(c10::div_floor_integer(x0, 8L))) + (64L*x1) + (64L*x1_inner) + (static_cast<long>(x0) % static_cast<long>(8L))), 8192L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((8L*(c10::div_floor_integer(x0, 8L))) + (64L*x1) + (64L*x1_inner) + (static_cast<long>(x0) % static_cast<long>(8L))), 64L)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(640L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-2L) + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<int>((-2L) + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>((-11520L) + x3 + (640L*x2) + (5120L*x1) + (40960L*x0)), to_float_mask(tmp10));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            tmp13.store(out_ptr1 + static_cast<long>(x3 + (640L*x2) + (7680L*x1) + (92160L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr1[static_cast<long>((640L*x0) + (92160L*(c10::div_floor_integer((x0 + (144L*x2) + (144L*x2_inner) + (11520L*x1)), 92160L))) + (static_cast<long>((x2 + x2_inner + (80L*x1))) % static_cast<long>(640L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (16L*x1) + (1024L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>((128L*x1) + (1024L*x2) + (8192L*(c10::div_floor_integer((x1 + (8L*x2) + (64L*x3) + (64L*x3_inner) + (1024L*x0)), 8192L))) + (static_cast<long>(c10::div_floor_integer((x1 + (8L*x2) + (64L*x3) + (64L*x3_inner) + (1024L*x0)), 64L)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (128L*x1) + (1024L*x0)));
                    }
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((128L*x1) + (8192L*(c10::div_floor_integer((x1 + (64L*x2) + (64L*x2_inner) + (1024L*x0)), 8192L))) + (static_cast<long>((x2 + x2_inner + (16L*x0))) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_bmm_mul_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (144L*x1) + (9216L*x0))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp3 = c10::convert<long>(11L + (23L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 12L)));
                            auto tmp4 = static_cast<long>(192);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(static_cast<long>((11L + (23L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 12L)))) % static_cast<long>(24L));
                                auto tmp8 = static_cast<long>(23);
                                auto tmp9 = tmp7 < tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_ptr1[static_cast<long>((23L*(c10::div_floor_integer((11L + (23L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 12L))), 24L))) + (184L*(static_cast<long>(x1) % static_cast<long>(8L))) + (1472L*x0) + (static_cast<long>((11L + (23L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 12L)))) % static_cast<long>(24L)))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp14 = c10::convert<long>(11L + (23L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(12L)));
                            auto tmp15 = tmp14 < tmp4;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(static_cast<long>((11L + (23L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(12L)))) % static_cast<long>(24L));
                                auto tmp18 = static_cast<long>(23);
                                auto tmp19 = tmp17 < tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = in_ptr2[static_cast<long>((23L*(static_cast<long>(c10::div_floor_integer((11L + (23L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(12L))), 24L)) % static_cast<long>(8L))) + (184L*(c10::div_floor_integer(x1, 8L))) + (1472L*x0) + (static_cast<long>((11L + (23L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(12L)))) % static_cast<long>(24L)))];
                                    return tmp21;
                                }
                                ;
                                auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp24 = decltype(tmp13)(tmp13 + tmp23);
                            auto tmp25 = decltype(tmp2)(tmp2 + tmp24);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp25);
                        }
                        out_ptr0[static_cast<long>(x1 + (64L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (144L*x1) + (9216L*x0))];
                        auto tmp26 = out_ptr0[static_cast<long>(x1 + (64L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = c10::convert<long>(11L + (23L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 12L)));
                        auto tmp4 = static_cast<long>(192);
                        auto tmp5 = tmp3 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = c10::convert<long>(static_cast<long>((11L + (23L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 12L)))) % static_cast<long>(24L));
                            auto tmp8 = static_cast<long>(23);
                            auto tmp9 = tmp7 < tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = in_ptr1[static_cast<long>((23L*(c10::div_floor_integer((11L + (23L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 12L))), 24L))) + (184L*(static_cast<long>(x1) % static_cast<long>(8L))) + (1472L*x0) + (static_cast<long>((11L + (23L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 12L)))) % static_cast<long>(24L)))];
                                return tmp11;
                            }
                            ;
                            auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp14 = c10::convert<long>(11L + (23L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(12L)));
                        auto tmp15 = tmp14 < tmp4;
                        auto tmp16 = [&]
                        {
                            auto tmp17 = c10::convert<long>(static_cast<long>((11L + (23L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(12L)))) % static_cast<long>(24L));
                            auto tmp18 = static_cast<long>(23);
                            auto tmp19 = tmp17 < tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = in_ptr2[static_cast<long>((23L*(static_cast<long>(c10::div_floor_integer((11L + (23L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(12L))), 24L)) % static_cast<long>(8L))) + (184L*(c10::div_floor_integer(x1, 8L))) + (1472L*x0) + (static_cast<long>((11L + (23L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(12L)))) % static_cast<long>(24L)))];
                                return tmp21;
                            }
                            ;
                            auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                        auto tmp24 = decltype(tmp13)(tmp13 + tmp23);
                        auto tmp25 = decltype(tmp2)(tmp2 + tmp24);
                        auto tmp27 = decltype(tmp25)(tmp25 - tmp26);
                        auto tmp28 = std::exp(tmp27);
                        in_out_ptr0[static_cast<long>(x2 + (144L*x1) + (9216L*x0))] = tmp28;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr3[static_cast<long>((640L*x0) + (92160L*(c10::div_floor_integer((2304L + x0 + (144L*x2) + (144L*x2_inner) + (11520L*x1)), 92160L))) + (static_cast<long>((16L + x2 + x2_inner + (80L*x1))) % static_cast<long>(640L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (4096L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((64L*x1) + (4096L*(c10::div_floor_integer((x2 + x2_inner), 64L))) + (32768L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
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
                        tmp16.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_mean_silu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x2) + (131072L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
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
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1 = args
    args.clear()
    assert_size_stride(arg0_1, (24, ), (1, ))
    assert_size_stride(arg1_1, (24, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, ), (1, ))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (256, ), (1, ))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (64, ), (1, ))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (256, ), (1, ))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (128, ), (1, ))
    assert_size_stride(arg22_1, (128, ), (1, ))
    assert_size_stride(arg23_1, (128, ), (1, ))
    assert_size_stride(arg24_1, (512, ), (1, ))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, ), (1, ))
    assert_size_stride(arg27_1, (512, ), (1, ))
    assert_size_stride(arg28_1, (128, ), (1, ))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (128, ), (1, ))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (512, ), (1, ))
    assert_size_stride(arg33_1, (512, ), (1, ))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (256, ), (1, ))
    assert_size_stride(arg37_1, (256, ), (1, ))
    assert_size_stride(arg38_1, (1024, ), (1, ))
    assert_size_stride(arg39_1, (1024, ), (1, ))
    assert_size_stride(arg40_1, (1024, ), (1, ))
    assert_size_stride(arg41_1, (1024, ), (1, ))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (23, 16), (16, 1))
    assert_size_stride(arg45_1, (23, 16), (16, 1))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (1024, ), (1, ))
    assert_size_stride(arg49_1, (1024, ), (1, ))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (23, 16), (16, 1))
    assert_size_stride(arg53_1, (23, 16), (16, 1))
    assert_size_stride(arg54_1, (512, ), (1, ))
    assert_size_stride(arg55_1, (512, ), (1, ))
    assert_size_stride(arg56_1, (2048, ), (1, ))
    assert_size_stride(arg57_1, (2048, ), (1, ))
    assert_size_stride(arg58_1, (2048, ), (1, ))
    assert_size_stride(arg59_1, (2048, ), (1, ))
    assert_size_stride(arg60_1, (512, ), (1, ))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (23, 16), (16, 1))
    assert_size_stride(arg63_1, (23, 16), (16, 1))
    assert_size_stride(arg64_1, (512, ), (1, ))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (2048, ), (1, ))
    assert_size_stride(arg67_1, (2048, ), (1, ))
    assert_size_stride(arg68_1, (24, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg69_1, (32, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(arg70_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg71_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg72_1, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg73_1, (1, 1, 3), (3, 3, 1))
    assert_size_stride(arg74_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg75_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg76_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg77_1, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg78_1, (1, 1, 3), (3, 3, 1))
    assert_size_stride(arg79_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg80_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg81_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg82_1, (1, 1, 5), (5, 5, 1))
    assert_size_stride(arg83_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg84_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg85_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg86_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg87_1, (1, 1, 5), (5, 5, 1))
    assert_size_stride(arg88_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg89_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg90_1, (256, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg91_1, (1, 1, 5), (5, 5, 1))
    assert_size_stride(arg92_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg93_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg94_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg95_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg96_1, (384, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg97_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg98_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg99_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg100_1, (640, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg101_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg102_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg103_1, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg104_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg105_1, (640, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg106_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg107_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg108_1, (1000, ), (1, ))
    assert_size_stride(arg109_1, (24, ), (1, ))
    assert_size_stride(arg110_1, (24, ), (1, ))
    assert_size_stride(arg111_1, (32, ), (1, ))
    assert_size_stride(arg112_1, (32, ), (1, ))
    assert_size_stride(arg113_1, (64, ), (1, ))
    assert_size_stride(arg114_1, (64, ), (1, ))
    assert_size_stride(arg115_1, (64, ), (1, ))
    assert_size_stride(arg116_1, (64, ), (1, ))
    assert_size_stride(arg117_1, (64, ), (1, ))
    assert_size_stride(arg118_1, (64, ), (1, ))
    assert_size_stride(arg119_1, (256, ), (1, ))
    assert_size_stride(arg120_1, (256, ), (1, ))
    assert_size_stride(arg121_1, (256, ), (1, ))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (64, ), (1, ))
    assert_size_stride(arg124_1, (64, ), (1, ))
    assert_size_stride(arg125_1, (64, ), (1, ))
    assert_size_stride(arg126_1, (64, ), (1, ))
    assert_size_stride(arg127_1, (256, ), (1, ))
    assert_size_stride(arg128_1, (256, ), (1, ))
    assert_size_stride(arg129_1, (128, ), (1, ))
    assert_size_stride(arg130_1, (128, ), (1, ))
    assert_size_stride(arg131_1, (128, ), (1, ))
    assert_size_stride(arg132_1, (128, ), (1, ))
    assert_size_stride(arg133_1, (512, ), (1, ))
    assert_size_stride(arg134_1, (512, ), (1, ))
    assert_size_stride(arg135_1, (512, ), (1, ))
    assert_size_stride(arg136_1, (512, ), (1, ))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (128, ), (1, ))
    assert_size_stride(arg139_1, (128, ), (1, ))
    assert_size_stride(arg140_1, (128, ), (1, ))
    assert_size_stride(arg141_1, (512, ), (1, ))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (256, ), (1, ))
    assert_size_stride(arg147_1, (1024, ), (1, ))
    assert_size_stride(arg148_1, (1024, ), (1, ))
    assert_size_stride(arg149_1, (1024, ), (1, ))
    assert_size_stride(arg150_1, (1024, ), (1, ))
    assert_size_stride(arg151_1, (256, ), (1, ))
    assert_size_stride(arg152_1, (256, ), (1, ))
    assert_size_stride(arg153_1, (256, ), (1, ))
    assert_size_stride(arg154_1, (256, ), (1, ))
    assert_size_stride(arg155_1, (1024, ), (1, ))
    assert_size_stride(arg156_1, (1024, ), (1, ))
    assert_size_stride(arg157_1, (512, ), (1, ))
    assert_size_stride(arg158_1, (512, ), (1, ))
    assert_size_stride(arg159_1, (512, ), (1, ))
    assert_size_stride(arg160_1, (512, ), (1, ))
    assert_size_stride(arg161_1, (2048, ), (1, ))
    assert_size_stride(arg162_1, (2048, ), (1, ))
    assert_size_stride(arg163_1, (2048, ), (1, ))
    assert_size_stride(arg164_1, (2048, ), (1, ))
    assert_size_stride(arg165_1, (512, ), (1, ))
    assert_size_stride(arg166_1, (512, ), (1, ))
    assert_size_stride(arg167_1, (512, ), (1, ))
    assert_size_stride(arg168_1, (512, ), (1, ))
    assert_size_stride(arg169_1, (2048, ), (1, ))
    assert_size_stride(arg170_1, (2048, ), (1, ))
    assert_size_stride(arg171_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    buf0 = empty_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((24, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg171_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg171_1
    del arg68_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 24, 128, 128), (393216, 1, 3072, 24))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = buf3; del buf3  # reuse
    buf5 = empty_strided((32, 24, 3, 3), (216, 1, 72, 24), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_silu_1(c_void_p(buf4.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(buf5.data_ptr()))
    del arg0_1
    del arg109_1
    del arg110_1
    del arg1_1
    del arg69_1
    # Source Nodes: [x_4, x_5], Original ATen: [aten.convolution, aten.silu]
    buf6 = extern_kernels.convolution(buf4, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf6, (8, 32, 128, 128), (524288, 1, 4096, 32))
    del buf4
    del buf5
    buf7 = buf6; del buf6  # reuse
    buf8 = buf7; del buf7  # reuse
    buf9 = empty_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_silu_2(c_void_p(buf8.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(buf9.data_ptr()))
    del arg111_1
    del arg112_1
    del arg2_1
    del arg3_1
    del arg70_1
    # Source Nodes: [x_10, x_9], Original ATen: [aten.convolution, aten.silu]
    buf10 = extern_kernels.convolution(buf8, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf10, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    del buf8
    buf11 = buf10; del buf10  # reuse
    buf12 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_silu_3(c_void_p(buf11.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf12.data_ptr()))
    del arg113_1
    del arg114_1
    del arg4_1
    del arg5_1
    del buf11
    # Source Nodes: [x_16], Original ATen: [aten.convolution]
    buf13 = extern_kernels.convolution(buf12, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf13, (8, 64, 64, 64), (262144, 1, 4096, 64))
    del arg71_1
    buf14 = buf13; del buf13  # reuse
    buf15 = buf14; del buf14  # reuse
    buf16 = empty_strided((64, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_silu_4(c_void_p(buf15.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(buf16.data_ptr()))
    del arg115_1
    del arg116_1
    del arg6_1
    del arg72_1
    del arg7_1
    # Source Nodes: [x_21, x_22], Original ATen: [aten.convolution, aten.silu]
    buf17 = extern_kernels.convolution(buf15, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
    assert_size_stride(buf17, (8, 64, 64, 64), (262144, 1, 4096, 64))
    del buf15
    buf18 = buf17; del buf17  # reuse
    buf19 = empty((8, 64), device='cpu', dtype=torch.float32)
    buf20 = buf19; del buf19  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_5(c_void_p(buf18.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()))
    del arg117_1
    del arg118_1
    del arg8_1
    del arg9_1
    # Source Nodes: [y_1], Original ATen: [aten.convolution]
    buf21 = extern_kernels.convolution(reinterpret_tensor(buf20, (8, 1, 64), (64, 0, 1), 0), arg73_1, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf21, (8, 1, 64), (64, 64, 1))
    del arg73_1
    del buf20
    buf22 = buf18; del buf18  # reuse
    cpp_fused_mul_silu_6(c_void_p(buf22.data_ptr()), c_void_p(buf21.data_ptr()))
    # Source Nodes: [x_27, x_29, x_30], Original ATen: [aten.convolution, aten.mul, aten.silu]
    buf23 = extern_kernels.convolution(buf22, arg74_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf23, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    del arg74_1
    del buf22
    # Source Nodes: [x_38], Original ATen: [aten.convolution]
    buf24 = extern_kernels.convolution(buf12, arg75_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf24, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    del arg75_1
    del buf12
    buf25 = buf23; del buf23  # reuse
    buf26 = buf25; del buf25  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_silu_7(c_void_p(buf26.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()))
    del arg10_1
    del arg119_1
    del arg11_1
    del arg120_1
    del arg121_1
    del arg122_1
    del arg12_1
    del arg13_1
    del buf24
    # Source Nodes: [shortcut_1, x_44], Original ATen: [aten.convolution, aten.silu]
    buf27 = extern_kernels.convolution(buf26, arg76_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf27, (8, 64, 64, 64), (262144, 1, 4096, 64))
    del arg76_1
    buf28 = buf27; del buf27  # reuse
    buf29 = buf28; del buf28  # reuse
    buf30 = buf16; del buf16  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_silu_8(c_void_p(buf29.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(buf30.data_ptr()))
    del arg123_1
    del arg124_1
    del arg14_1
    del arg15_1
    del arg77_1
    # Source Nodes: [x_49, x_50], Original ATen: [aten.convolution, aten.silu]
    buf31 = extern_kernels.convolution(buf29, buf30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
    assert_size_stride(buf31, (8, 64, 64, 64), (262144, 1, 4096, 64))
    del buf29
    del buf30
    buf32 = buf31; del buf31  # reuse
    buf33 = reinterpret_tensor(buf21, (8, 64), (64, 1), 0); del buf21  # reuse
    buf34 = buf33; del buf33  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_9(c_void_p(buf32.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()))
    del arg125_1
    del arg126_1
    del arg16_1
    del arg17_1
    # Source Nodes: [y_4], Original ATen: [aten.convolution]
    buf35 = extern_kernels.convolution(reinterpret_tensor(buf34, (8, 1, 64), (64, 0, 1), 0), arg78_1, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf35, (8, 1, 64), (64, 64, 1))
    del arg78_1
    del buf34
    buf36 = buf32; del buf32  # reuse
    cpp_fused_mul_silu_10(c_void_p(buf36.data_ptr()), c_void_p(buf35.data_ptr()))
    del buf35
    # Source Nodes: [x_55, x_57, x_58], Original ATen: [aten.convolution, aten.mul, aten.silu]
    buf37 = extern_kernels.convolution(buf36, arg79_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf37, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    del arg79_1
    del buf36
    buf38 = buf26; del buf26  # reuse
    buf39 = buf38; del buf38  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_silu_11(c_void_p(buf39.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()))
    del arg127_1
    del arg128_1
    del arg18_1
    del arg19_1
    del buf37
    # Source Nodes: [shortcut_2, x_67], Original ATen: [aten.convolution, aten.silu]
    buf40 = extern_kernels.convolution(buf39, arg80_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf40, (8, 128, 64, 64), (524288, 1, 8192, 128))
    del arg80_1
    buf41 = buf40; del buf40  # reuse
    buf42 = buf41; del buf41  # reuse
    buf43 = reinterpret_tensor(buf9, (128, 16, 3, 3), (144, 1, 48, 16), 0); del buf9  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_silu_12(c_void_p(buf42.data_ptr()), c_void_p(arg129_1.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(buf43.data_ptr()))
    del arg129_1
    del arg130_1
    del arg20_1
    del arg21_1
    del arg81_1
    # Source Nodes: [x_72, x_73], Original ATen: [aten.convolution, aten.silu]
    buf44 = extern_kernels.convolution(buf42, buf43, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf44, (8, 128, 32, 32), (131072, 1, 4096, 128))
    del buf42
    buf45 = buf44; del buf44  # reuse
    buf46 = empty((8, 128), device='cpu', dtype=torch.float32)
    buf47 = buf46; del buf46  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_13(c_void_p(buf45.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()))
    del arg131_1
    del arg132_1
    del arg22_1
    del arg23_1
    # Source Nodes: [y_7], Original ATen: [aten.convolution]
    buf48 = extern_kernels.convolution(reinterpret_tensor(buf47, (8, 1, 128), (128, 0, 1), 0), arg82_1, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf48, (8, 1, 128), (128, 128, 1))
    del arg82_1
    del buf47
    buf49 = buf45; del buf45  # reuse
    cpp_fused_mul_silu_14(c_void_p(buf49.data_ptr()), c_void_p(buf48.data_ptr()))
    # Source Nodes: [x_78, x_80, x_81], Original ATen: [aten.convolution, aten.mul, aten.silu]
    buf50 = extern_kernels.convolution(buf49, arg83_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf50, (8, 512, 32, 32), (524288, 1, 16384, 512))
    del arg83_1
    del buf49
    # Source Nodes: [x_89], Original ATen: [aten.convolution]
    buf51 = extern_kernels.convolution(buf39, arg84_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf51, (8, 512, 32, 32), (524288, 1, 16384, 512))
    del arg84_1
    del buf39
    buf52 = buf50; del buf50  # reuse
    buf53 = buf52; del buf52  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_silu_15(c_void_p(buf53.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(arg135_1.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()))
    del arg133_1
    del arg134_1
    del arg135_1
    del arg136_1
    del arg24_1
    del arg25_1
    del arg26_1
    del arg27_1
    del buf51
    # Source Nodes: [shortcut_3, x_95], Original ATen: [aten.convolution, aten.silu]
    buf54 = extern_kernels.convolution(buf53, arg85_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf54, (8, 128, 32, 32), (131072, 1, 4096, 128))
    del arg85_1
    buf55 = buf54; del buf54  # reuse
    buf56 = buf55; del buf55  # reuse
    buf57 = buf43; del buf43  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_silu_16(c_void_p(buf56.data_ptr()), c_void_p(arg137_1.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(buf57.data_ptr()))
    del arg137_1
    del arg138_1
    del arg28_1
    del arg29_1
    del arg86_1
    # Source Nodes: [x_100, x_101], Original ATen: [aten.convolution, aten.silu]
    buf58 = extern_kernels.convolution(buf56, buf57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf58, (8, 128, 32, 32), (131072, 1, 4096, 128))
    del buf56
    del buf57
    buf59 = buf58; del buf58  # reuse
    buf60 = reinterpret_tensor(buf48, (8, 128), (128, 1), 0); del buf48  # reuse
    buf61 = buf60; del buf60  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_17(c_void_p(buf59.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()))
    del arg139_1
    del arg140_1
    del arg30_1
    del arg31_1
    # Source Nodes: [y_10], Original ATen: [aten.convolution]
    buf62 = extern_kernels.convolution(reinterpret_tensor(buf61, (8, 1, 128), (128, 0, 1), 0), arg87_1, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf62, (8, 1, 128), (128, 128, 1))
    del arg87_1
    del buf61
    buf63 = buf59; del buf59  # reuse
    cpp_fused_mul_silu_18(c_void_p(buf63.data_ptr()), c_void_p(buf62.data_ptr()))
    del buf62
    # Source Nodes: [x_106, x_108, x_109], Original ATen: [aten.convolution, aten.mul, aten.silu]
    buf64 = extern_kernels.convolution(buf63, arg88_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf64, (8, 512, 32, 32), (524288, 1, 16384, 512))
    del arg88_1
    del buf63
    buf65 = buf53; del buf53  # reuse
    buf66 = buf65; del buf65  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_silu_19(c_void_p(buf66.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()))
    del arg141_1
    del arg142_1
    del arg32_1
    del arg33_1
    del buf64
    # Source Nodes: [shortcut_4, x_118], Original ATen: [aten.convolution, aten.silu]
    buf67 = extern_kernels.convolution(buf66, arg89_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf67, (8, 256, 32, 32), (262144, 1, 8192, 256))
    del arg89_1
    buf68 = buf67; del buf67  # reuse
    buf69 = buf68; del buf68  # reuse
    buf70 = empty_strided((256, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_silu_20(c_void_p(buf69.data_ptr()), c_void_p(arg143_1.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(buf70.data_ptr()))
    del arg143_1
    del arg144_1
    del arg34_1
    del arg35_1
    del arg90_1
    # Source Nodes: [x_123, x_124], Original ATen: [aten.convolution, aten.silu]
    buf71 = extern_kernels.convolution(buf69, buf70, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
    assert_size_stride(buf71, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del buf69
    del buf70
    buf72 = buf71; del buf71  # reuse
    buf73 = empty((8, 256), device='cpu', dtype=torch.float32)
    buf74 = buf73; del buf73  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_21(c_void_p(buf72.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()))
    del arg145_1
    del arg146_1
    del arg36_1
    del arg37_1
    # Source Nodes: [y_13], Original ATen: [aten.convolution]
    buf75 = extern_kernels.convolution(reinterpret_tensor(buf74, (8, 1, 256), (256, 0, 1), 0), arg91_1, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf75, (8, 1, 256), (256, 256, 1))
    del arg91_1
    del buf74
    buf76 = buf72; del buf72  # reuse
    cpp_fused_mul_silu_22(c_void_p(buf76.data_ptr()), c_void_p(buf75.data_ptr()))
    del buf75
    # Source Nodes: [x_129, x_131, x_132], Original ATen: [aten.convolution, aten.mul, aten.silu]
    buf77 = extern_kernels.convolution(buf76, arg92_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf77, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg92_1
    # Source Nodes: [x_140], Original ATen: [aten.convolution]
    buf78 = extern_kernels.convolution(buf66, arg93_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf78, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg93_1
    del buf66
    buf79 = buf77; del buf77  # reuse
    buf80 = buf79; del buf79  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_silu_23(c_void_p(buf80.data_ptr()), c_void_p(arg147_1.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(arg149_1.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()))
    del arg147_1
    del arg148_1
    del arg149_1
    del arg150_1
    del arg38_1
    del arg39_1
    del arg40_1
    del arg41_1
    del buf78
    # Source Nodes: [shortcut_5, x_146], Original ATen: [aten.convolution, aten.silu]
    buf81 = extern_kernels.convolution(buf80, arg94_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf81, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg94_1
    buf82 = buf81; del buf81  # reuse
    buf83 = buf82; del buf82  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_24(c_void_p(buf83.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(arg152_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()))
    del arg151_1
    del arg152_1
    del arg42_1
    del arg43_1
    # Source Nodes: [kv, x_151], Original ATen: [aten.convolution, aten.silu]
    buf84 = extern_kernels.convolution(buf83, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf84, (8, 384, 16, 16), (98304, 1, 6144, 384))
    del arg96_1
    # Source Nodes: [q], Original ATen: [aten.convolution]
    buf85 = extern_kernels.convolution(buf83, arg95_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf85, (8, 128, 16, 16), (32768, 1, 2048, 128))
    del arg95_1
    buf86 = empty((64, 4, 64, 16), device='cpu', dtype=torch.float32)
    buf91 = empty((64, 4, 64, 16), device='cpu', dtype=torch.float32)
    buf87 = empty((64, 4, 16, 144), device='cpu', dtype=torch.float32)
    cpp_fused_clone_25(c_void_p(buf85.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf87.data_ptr()))
    buf88 = empty((256, 64, 144), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf86, (256, 64, 16), (1024, 16, 1), 0), reinterpret_tensor(buf87, (256, 16, 144), (2304, 144, 1), 0), out=buf88)
    buf89 = reinterpret_tensor(buf86, (256, 8, 8, 16), (1024, 128, 16, 1), 0); del buf86  # reuse
    cpp_fused_clone_26(c_void_p(buf85.data_ptr()), c_void_p(buf89.data_ptr()))
    del buf85
    buf90 = empty((16384, 23), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_157], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf89, (16384, 16), (16, 1), 0), reinterpret_tensor(arg45_1, (16, 23), (1, 16), 0), out=buf90)
    del arg45_1
    buf92 = empty((16384, 23), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_153], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf91, (16384, 16), (16, 1), 0), reinterpret_tensor(arg44_1, (16, 23), (1, 16), 0), out=buf92)
    del arg44_1
    buf93 = empty_strided((64, 4, 64, 1), (256, 64, 1, 16384), device='cpu', dtype=torch.float32)
    buf94 = reinterpret_tensor(buf88, (64, 4, 64, 144), (36864, 9216, 144, 1), 0); del buf88  # reuse
    buf95 = empty_strided((64, 4, 64, 1), (256, 64, 1, 16384), device='cpu', dtype=torch.float32)
    buf96 = buf94; del buf94  # reuse
    buf97 = empty((64, 4, 144, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_clone_mul_27(c_void_p(buf96.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf97.data_ptr()))
    del buf84
    del buf90
    del buf92
    del buf93
    buf98 = reinterpret_tensor(buf83, (256, 64, 32), (2048, 32, 1), 0); del buf83  # reuse
    # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf96, (256, 64, 144), (9216, 144, 1), 0), reinterpret_tensor(buf97, (256, 144, 32), (4608, 32, 1), 0), out=buf98)
    del buf97
    buf99 = buf76; del buf76  # reuse
    buf100 = buf99; del buf99  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_28(c_void_p(buf100.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(arg153_1.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()))
    del arg153_1
    del arg154_1
    del arg46_1
    del arg47_1
    del buf98
    # Source Nodes: [x_165, x_166], Original ATen: [aten.convolution, aten.silu]
    buf101 = extern_kernels.convolution(buf100, arg97_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf101, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg97_1
    del buf100
    buf102 = buf101; del buf101  # reuse
    buf103 = buf102; del buf102  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_silu_29(c_void_p(buf103.data_ptr()), c_void_p(arg155_1.data_ptr()), c_void_p(arg156_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(buf80.data_ptr()))
    del arg155_1
    del arg156_1
    del arg48_1
    del arg49_1
    del buf80
    # Source Nodes: [shortcut_6, x_174], Original ATen: [aten.convolution, aten.silu]
    buf104 = extern_kernels.convolution(buf103, arg98_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf104, (8, 512, 16, 16), (131072, 1, 8192, 512))
    del arg98_1
    buf105 = buf104; del buf104  # reuse
    buf106 = buf105; del buf105  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_30(c_void_p(buf106.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()))
    del arg157_1
    del arg158_1
    del arg50_1
    del arg51_1
    # Source Nodes: [kv_3, x_179], Original ATen: [aten.convolution, aten.silu]
    buf107 = extern_kernels.convolution(buf106, arg100_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf107, (8, 640, 16, 16), (163840, 1, 10240, 640))
    del arg100_1
    # Source Nodes: [q_5], Original ATen: [aten.convolution]
    buf108 = extern_kernels.convolution(buf106, arg99_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf108, (8, 128, 8, 8), (8192, 1, 1024, 128))
    del arg99_1
    del buf106
    buf109 = empty((64, 4, 16, 16), device='cpu', dtype=torch.float32)
    buf114 = empty((64, 4, 16, 16), device='cpu', dtype=torch.float32)
    buf110 = buf87; del buf87  # reuse
    cpp_fused_clone_31(c_void_p(buf108.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf110.data_ptr()))
    buf111 = empty((256, 16, 144), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf109, (256, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf110, (256, 16, 144), (2304, 144, 1), 0), out=buf111)
    buf112 = reinterpret_tensor(buf109, (256, 4, 4, 16), (256, 64, 16, 1), 0); del buf109  # reuse
    cpp_fused_clone_32(c_void_p(buf108.data_ptr()), c_void_p(buf112.data_ptr()))
    del buf108
    buf113 = empty((4096, 23), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_185], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf112, (4096, 16), (16, 1), 0), reinterpret_tensor(arg53_1, (16, 23), (1, 16), 0), out=buf113)
    del arg53_1
    del buf112
    buf115 = empty((4096, 23), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_181], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf114, (4096, 16), (16, 1), 0), reinterpret_tensor(arg52_1, (16, 23), (1, 16), 0), out=buf115)
    del arg52_1
    buf116 = empty_strided((64, 4, 16, 1), (64, 16, 1, 4096), device='cpu', dtype=torch.float32)
    buf117 = reinterpret_tensor(buf111, (64, 4, 16, 144), (9216, 2304, 144, 1), 0); del buf111  # reuse
    buf118 = empty_strided((64, 4, 16, 1), (64, 16, 1, 4096), device='cpu', dtype=torch.float32)
    buf119 = buf117; del buf117  # reuse
    buf120 = reinterpret_tensor(buf96, (64, 4, 144, 64), (36864, 9216, 64, 1), 0); del buf96  # reuse
    cpp_fused__softmax_add_clone_mul_33(c_void_p(buf119.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf120.data_ptr()))
    del buf107
    buf121 = reinterpret_tensor(buf91, (256, 16, 64), (1024, 64, 1), 0); del buf91  # reuse
    # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf119, (256, 16, 144), (2304, 144, 1), 0), reinterpret_tensor(buf120, (256, 144, 64), (9216, 64, 1), 0), out=buf121)
    del buf120
    buf122 = reinterpret_tensor(buf89, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf89  # reuse
    buf123 = buf122; del buf122  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_34(c_void_p(buf123.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(arg159_1.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()))
    del arg159_1
    del arg160_1
    del arg54_1
    del arg55_1
    del buf121
    # Source Nodes: [x_193, x_194], Original ATen: [aten.convolution, aten.silu]
    buf124 = extern_kernels.convolution(buf123, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf124, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    del arg101_1
    # Source Nodes: [x_201], Original ATen: [aten.convolution]
    buf125 = extern_kernels.convolution(buf103, arg102_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf125, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    del arg102_1
    del buf103
    buf126 = buf124; del buf124  # reuse
    buf127 = buf126; del buf126  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_silu_35(c_void_p(buf127.data_ptr()), c_void_p(arg161_1.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()))
    del arg161_1
    del arg162_1
    del arg163_1
    del arg164_1
    del arg56_1
    del arg57_1
    del arg58_1
    del arg59_1
    del buf125
    # Source Nodes: [shortcut_7, x_207], Original ATen: [aten.convolution, aten.silu]
    buf128 = extern_kernels.convolution(buf127, arg103_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf128, (8, 512, 8, 8), (32768, 1, 4096, 512))
    del arg103_1
    buf129 = buf128; del buf128  # reuse
    buf130 = buf129; del buf129  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_36(c_void_p(buf130.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()))
    del arg165_1
    del arg166_1
    del arg60_1
    del arg61_1
    # Source Nodes: [kv_6, x_212], Original ATen: [aten.convolution, aten.silu]
    buf131 = extern_kernels.convolution(buf130, arg105_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf131, (8, 640, 8, 8), (40960, 1, 5120, 640))
    del arg105_1
    # Source Nodes: [q_10], Original ATen: [aten.convolution]
    buf132 = extern_kernels.convolution(buf130, arg104_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf132, (8, 128, 8, 8), (8192, 1, 1024, 128))
    del arg104_1
    buf133 = reinterpret_tensor(buf114, (64, 64, 16), (16, 1024, 1), 0); del buf114  # reuse
    buf134 = empty_strided((8, 640, 12, 12), (92160, 1, 7680, 640), device='cpu', dtype=torch.float32)
    buf135 = empty_strided((64, 16, 144), (16, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_bmm_constant_pad_nd_37(c_void_p(buf132.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()))
    del buf131
    buf136 = reinterpret_tensor(buf119, (64, 64, 144), (9216, 144, 1), 0); del buf119  # reuse
    # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf133, buf135, out=buf136)
    del buf135
    buf137 = reinterpret_tensor(buf133, (64, 8, 8, 16), (1024, 128, 16, 1), 0); del buf133  # reuse
    cpp_fused_clone_38(c_void_p(buf132.data_ptr()), c_void_p(buf137.data_ptr()))
    buf138 = buf115; del buf115  # reuse
    # Source Nodes: [x_218], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf137, (4096, 16), (16, 1), 0), reinterpret_tensor(arg63_1, (16, 23), (1, 16), 0), out=buf138)
    del arg63_1
    buf139 = buf137; del buf137  # reuse
    cpp_fused_clone_39(c_void_p(buf132.data_ptr()), c_void_p(buf139.data_ptr()))
    del buf132
    buf140 = buf113; del buf113  # reuse
    # Source Nodes: [x_214], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf139, (4096, 16), (16, 1), 0), reinterpret_tensor(arg62_1, (16, 23), (1, 16), 0), out=buf140)
    del arg62_1
    del buf139
    buf141 = reinterpret_tensor(buf118, (64, 1, 64, 1), (64, 4096, 1, 4096), 0); del buf118  # reuse
    buf142 = reinterpret_tensor(buf136, (64, 1, 64, 144), (9216, 589824, 144, 1), 0); del buf136  # reuse
    buf143 = reinterpret_tensor(buf116, (64, 1, 64, 1), (64, 4096, 1, 4096), 0); del buf116  # reuse
    buf144 = reinterpret_tensor(buf142, (64, 1, 64, 144), (9216, 9216, 144, 1), 0); del buf142  # reuse
    buf145 = reinterpret_tensor(buf110, (64, 144, 64), (64, 4096, 1), 0); del buf110  # reuse
    cpp_fused__softmax_add_bmm_mul_40(c_void_p(buf144.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf145.data_ptr()))
    del buf134
    del buf138
    del buf140
    del buf141
    del buf143
    buf146 = reinterpret_tensor(buf130, (64, 64, 64), (4096, 64, 1), 0); del buf130  # reuse
    # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf144, (64, 64, 144), (9216, 144, 1), 0), buf145, out=buf146)
    del buf144
    del buf145
    buf147 = buf123; del buf123  # reuse
    buf148 = buf147; del buf147  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_41(c_void_p(buf148.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(arg167_1.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()))
    del arg167_1
    del arg168_1
    del arg64_1
    del arg65_1
    del buf146
    # Source Nodes: [x_226, x_227], Original ATen: [aten.convolution, aten.silu]
    buf149 = extern_kernels.convolution(buf148, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf149, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    del arg106_1
    del buf148
    buf150 = buf127; del buf127  # reuse
    buf151 = reinterpret_tensor(buf95, (8, 2048, 1, 1), (2048, 1, 16384, 16384), 0); del buf95  # reuse
    buf152 = reinterpret_tensor(buf151, (8, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf151  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_mean_silu_42(c_void_p(buf150.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(arg169_1.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()))
    del arg169_1
    del arg170_1
    del arg66_1
    del arg67_1
    del buf149
    del buf150
    buf153 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_242], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg108_1, reinterpret_tensor(buf152, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg107_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf153)
    del arg107_1
    del arg108_1
    return (buf153, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((23, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((23, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((23, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((23, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((23, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((23, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((32, 24, 3, 3), (216, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((1, 1, 3), (3, 3, 1), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((1, 1, 3), (3, 3, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((1, 1, 5), (5, 5, 1), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((1, 1, 5), (5, 5, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((256, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((1, 1, 5), (5, 5, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((384, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((640, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((640, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('eca_halonext26ts', benchmark_compiled_module)
