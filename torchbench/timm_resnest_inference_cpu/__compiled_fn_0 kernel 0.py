
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x2) + (401408L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(64L + x1 + (128L*x2) + (401408L*x0)));
                            auto tmp2 = tmp0 + tmp1;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(3136.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
            }
        }
    }
}
''')


cpp_fused__softmax_mul_sum_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(64L + x2 + (128L*x0)));
                    auto tmp3 = at::vec::maximum(tmp1, tmp2);
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.exp();
                    tmp5.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (128L*x0)));
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (401408L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (128L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(64L + x2 + (128L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x2 + (128L*x1) + (401408L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (200704L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x2) + (802816L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(128L + x1 + (256L*x2) + (802816L*x0)));
                            auto tmp2 = tmp0 + tmp1;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(3136.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__softmax_avg_pool2d_mul_sum_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x2 + (256L*x0)));
                    auto tmp3 = at::vec::maximum(tmp1, tmp2);
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.exp();
                    tmp5.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (256L*x0)));
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (802816L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (256L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(128L + x2 + (256L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (256L*x1) + (802816L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (401408L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-1L) + (2L*x1));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(56);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<int>((-1L) + (2L*x2));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(out_ptr1 + static_cast<long>((-7296L) + x3 + (256L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp10));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            auto tmp14 = c10::convert<int>(2L*x2);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = masked_load(out_ptr1 + static_cast<long>((-7168L) + x3 + (256L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp18));
                                return tmp20;
                            }
                            ;
                            auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp18));
                            auto tmp22 = tmp21 + tmp13;
                            auto tmp23 = c10::convert<int>(1L + (2L*x2));
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = masked_load(out_ptr1 + static_cast<long>((-7040L) + x3 + (256L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp27));
                                return tmp29;
                            }
                            ;
                            auto tmp30 = decltype(tmp28())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp28(), to_float_mask(tmp27));
                            auto tmp31 = tmp30 + tmp22;
                            auto tmp32 = c10::convert<int>(2L*x1);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = masked_load(out_ptr1 + static_cast<long>((-128L) + x3 + (256L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp36));
                                return tmp38;
                            }
                            ;
                            auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp37(), to_float_mask(tmp36));
                            auto tmp40 = tmp39 + tmp31;
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(out_ptr1 + static_cast<long>(x3 + (256L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp41));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp41));
                            auto tmp45 = tmp44 + tmp40;
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = masked_load(out_ptr1 + static_cast<long>(128L + x3 + (256L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp46));
                                return tmp48;
                            }
                            ;
                            auto tmp49 = decltype(tmp47())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp47(), to_float_mask(tmp46));
                            auto tmp50 = tmp49 + tmp45;
                            auto tmp51 = c10::convert<int>(1L + (2L*x1));
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = masked_load(out_ptr1 + static_cast<long>(7040L + x3 + (256L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp55));
                                return tmp57;
                            }
                            ;
                            auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp56(), to_float_mask(tmp55));
                            auto tmp59 = tmp58 + tmp50;
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = masked_load(out_ptr1 + static_cast<long>(7168L + x3 + (256L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp60));
                                return tmp62;
                            }
                            ;
                            auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp61(), to_float_mask(tmp60));
                            auto tmp64 = tmp63 + tmp59;
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = masked_load(out_ptr1 + static_cast<long>(7296L + x3 + (256L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp65));
                                return tmp67;
                            }
                            ;
                            auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp66(), to_float_mask(tmp65));
                            auto tmp69 = tmp68 + tmp64;
                            auto tmp70 = static_cast<int>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<int>(57);
                            auto tmp73 = tmp0 < tmp72;
                            auto tmp74 = tmp71 & tmp73;
                            auto tmp75 = tmp6 >= tmp70;
                            auto tmp76 = tmp6 < tmp72;
                            auto tmp77 = tmp75 & tmp76;
                            auto tmp78 = tmp74 & tmp77;
                            auto tmp79 = [&]
                            {
                                auto tmp80 = c10::convert<int>((-1L) + (2L*x1));
                                auto tmp81 = static_cast<int>(0);
                                auto tmp82 = tmp80 >= tmp81;
                                auto tmp83 = static_cast<int>(56);
                                auto tmp84 = tmp80 < tmp83;
                                auto tmp85 = tmp82 & tmp84;
                                auto tmp86 = c10::convert<int>((-1L) + (2L*x2));
                                auto tmp87 = tmp86 >= tmp81;
                                auto tmp88 = tmp86 < tmp83;
                                auto tmp89 = tmp87 & tmp88;
                                auto tmp90 = tmp85 & tmp89;
                                auto tmp92 = tmp90 & tmp78;
                                auto tmp91 = [&]
                                {
                                    auto tmp93 = static_cast<float>(1.0);
                                    return tmp93;
                                }
                                ;
                                auto tmp94 = tmp90 ? tmp91() : static_cast<float>(1.0);
                                return tmp94;
                            }
                            ;
                            auto tmp95 = tmp78 ? tmp79() : static_cast<float>(0.0);
                            auto tmp96 = tmp14 >= tmp70;
                            auto tmp97 = tmp14 < tmp72;
                            auto tmp98 = tmp96 & tmp97;
                            auto tmp99 = tmp74 & tmp98;
                            auto tmp100 = [&]
                            {
                                auto tmp101 = c10::convert<int>((-1L) + (2L*x1));
                                auto tmp102 = static_cast<int>(0);
                                auto tmp103 = tmp101 >= tmp102;
                                auto tmp104 = static_cast<int>(56);
                                auto tmp105 = tmp101 < tmp104;
                                auto tmp106 = tmp103 & tmp105;
                                auto tmp107 = c10::convert<int>(2L*x2);
                                auto tmp108 = tmp107 >= tmp102;
                                auto tmp109 = tmp107 < tmp104;
                                auto tmp110 = tmp108 & tmp109;
                                auto tmp111 = tmp106 & tmp110;
                                auto tmp113 = tmp111 & tmp99;
                                auto tmp112 = [&]
                                {
                                    auto tmp114 = static_cast<float>(1.0);
                                    return tmp114;
                                }
                                ;
                                auto tmp115 = tmp111 ? tmp112() : static_cast<float>(1.0);
                                return tmp115;
                            }
                            ;
                            auto tmp116 = tmp99 ? tmp100() : static_cast<float>(0.0);
                            auto tmp117 = decltype(tmp116)(tmp116 + tmp95);
                            auto tmp118 = tmp23 >= tmp70;
                            auto tmp119 = tmp23 < tmp72;
                            auto tmp120 = tmp118 & tmp119;
                            auto tmp121 = tmp74 & tmp120;
                            auto tmp122 = [&]
                            {
                                auto tmp123 = c10::convert<int>((-1L) + (2L*x1));
                                auto tmp124 = static_cast<int>(0);
                                auto tmp125 = tmp123 >= tmp124;
                                auto tmp126 = static_cast<int>(56);
                                auto tmp127 = tmp123 < tmp126;
                                auto tmp128 = tmp125 & tmp127;
                                auto tmp129 = c10::convert<int>(1L + (2L*x2));
                                auto tmp130 = tmp129 >= tmp124;
                                auto tmp131 = tmp129 < tmp126;
                                auto tmp132 = tmp130 & tmp131;
                                auto tmp133 = tmp128 & tmp132;
                                auto tmp135 = tmp133 & tmp121;
                                auto tmp134 = [&]
                                {
                                    auto tmp136 = static_cast<float>(1.0);
                                    return tmp136;
                                }
                                ;
                                auto tmp137 = tmp133 ? tmp134() : static_cast<float>(1.0);
                                return tmp137;
                            }
                            ;
                            auto tmp138 = tmp121 ? tmp122() : static_cast<float>(0.0);
                            auto tmp139 = decltype(tmp138)(tmp138 + tmp117);
                            auto tmp140 = tmp32 >= tmp70;
                            auto tmp141 = tmp32 < tmp72;
                            auto tmp142 = tmp140 & tmp141;
                            auto tmp143 = tmp142 & tmp77;
                            auto tmp144 = [&]
                            {
                                auto tmp145 = c10::convert<int>(2L*x1);
                                auto tmp146 = static_cast<int>(0);
                                auto tmp147 = tmp145 >= tmp146;
                                auto tmp148 = static_cast<int>(56);
                                auto tmp149 = tmp145 < tmp148;
                                auto tmp150 = tmp147 & tmp149;
                                auto tmp151 = c10::convert<int>((-1L) + (2L*x2));
                                auto tmp152 = tmp151 >= tmp146;
                                auto tmp153 = tmp151 < tmp148;
                                auto tmp154 = tmp152 & tmp153;
                                auto tmp155 = tmp150 & tmp154;
                                auto tmp157 = tmp155 & tmp143;
                                auto tmp156 = [&]
                                {
                                    auto tmp158 = static_cast<float>(1.0);
                                    return tmp158;
                                }
                                ;
                                auto tmp159 = tmp155 ? tmp156() : static_cast<float>(1.0);
                                return tmp159;
                            }
                            ;
                            auto tmp160 = tmp143 ? tmp144() : static_cast<float>(0.0);
                            auto tmp161 = decltype(tmp160)(tmp160 + tmp139);
                            auto tmp162 = tmp142 & tmp98;
                            auto tmp163 = [&]
                            {
                                auto tmp164 = c10::convert<int>(2L*x1);
                                auto tmp165 = static_cast<int>(0);
                                auto tmp166 = tmp164 >= tmp165;
                                auto tmp167 = static_cast<int>(56);
                                auto tmp168 = tmp164 < tmp167;
                                auto tmp169 = tmp166 & tmp168;
                                auto tmp170 = c10::convert<int>(2L*x2);
                                auto tmp171 = tmp170 >= tmp165;
                                auto tmp172 = tmp170 < tmp167;
                                auto tmp173 = tmp171 & tmp172;
                                auto tmp174 = tmp169 & tmp173;
                                auto tmp176 = tmp174 & tmp162;
                                auto tmp175 = [&]
                                {
                                    auto tmp177 = static_cast<float>(1.0);
                                    return tmp177;
                                }
                                ;
                                auto tmp178 = tmp174 ? tmp175() : static_cast<float>(1.0);
                                return tmp178;
                            }
                            ;
                            auto tmp179 = tmp162 ? tmp163() : static_cast<float>(0.0);
                            auto tmp180 = decltype(tmp179)(tmp179 + tmp161);
                            auto tmp181 = tmp142 & tmp120;
                            auto tmp182 = [&]
                            {
                                auto tmp183 = c10::convert<int>(2L*x1);
                                auto tmp184 = static_cast<int>(0);
                                auto tmp185 = tmp183 >= tmp184;
                                auto tmp186 = static_cast<int>(56);
                                auto tmp187 = tmp183 < tmp186;
                                auto tmp188 = tmp185 & tmp187;
                                auto tmp189 = c10::convert<int>(1L + (2L*x2));
                                auto tmp190 = tmp189 >= tmp184;
                                auto tmp191 = tmp189 < tmp186;
                                auto tmp192 = tmp190 & tmp191;
                                auto tmp193 = tmp188 & tmp192;
                                auto tmp195 = tmp193 & tmp181;
                                auto tmp194 = [&]
                                {
                                    auto tmp196 = static_cast<float>(1.0);
                                    return tmp196;
                                }
                                ;
                                auto tmp197 = tmp193 ? tmp194() : static_cast<float>(1.0);
                                return tmp197;
                            }
                            ;
                            auto tmp198 = tmp181 ? tmp182() : static_cast<float>(0.0);
                            auto tmp199 = decltype(tmp198)(tmp198 + tmp180);
                            auto tmp200 = tmp51 >= tmp70;
                            auto tmp201 = tmp51 < tmp72;
                            auto tmp202 = tmp200 & tmp201;
                            auto tmp203 = tmp202 & tmp77;
                            auto tmp204 = [&]
                            {
                                auto tmp205 = c10::convert<int>(1L + (2L*x1));
                                auto tmp206 = static_cast<int>(0);
                                auto tmp207 = tmp205 >= tmp206;
                                auto tmp208 = static_cast<int>(56);
                                auto tmp209 = tmp205 < tmp208;
                                auto tmp210 = tmp207 & tmp209;
                                auto tmp211 = c10::convert<int>((-1L) + (2L*x2));
                                auto tmp212 = tmp211 >= tmp206;
                                auto tmp213 = tmp211 < tmp208;
                                auto tmp214 = tmp212 & tmp213;
                                auto tmp215 = tmp210 & tmp214;
                                auto tmp217 = tmp215 & tmp203;
                                auto tmp216 = [&]
                                {
                                    auto tmp218 = static_cast<float>(1.0);
                                    return tmp218;
                                }
                                ;
                                auto tmp219 = tmp215 ? tmp216() : static_cast<float>(1.0);
                                return tmp219;
                            }
                            ;
                            auto tmp220 = tmp203 ? tmp204() : static_cast<float>(0.0);
                            auto tmp221 = decltype(tmp220)(tmp220 + tmp199);
                            auto tmp222 = tmp202 & tmp98;
                            auto tmp223 = [&]
                            {
                                auto tmp224 = c10::convert<int>(1L + (2L*x1));
                                auto tmp225 = static_cast<int>(0);
                                auto tmp226 = tmp224 >= tmp225;
                                auto tmp227 = static_cast<int>(56);
                                auto tmp228 = tmp224 < tmp227;
                                auto tmp229 = tmp226 & tmp228;
                                auto tmp230 = c10::convert<int>(2L*x2);
                                auto tmp231 = tmp230 >= tmp225;
                                auto tmp232 = tmp230 < tmp227;
                                auto tmp233 = tmp231 & tmp232;
                                auto tmp234 = tmp229 & tmp233;
                                auto tmp236 = tmp234 & tmp222;
                                auto tmp235 = [&]
                                {
                                    auto tmp237 = static_cast<float>(1.0);
                                    return tmp237;
                                }
                                ;
                                auto tmp238 = tmp234 ? tmp235() : static_cast<float>(1.0);
                                return tmp238;
                            }
                            ;
                            auto tmp239 = tmp222 ? tmp223() : static_cast<float>(0.0);
                            auto tmp240 = decltype(tmp239)(tmp239 + tmp221);
                            auto tmp241 = tmp202 & tmp120;
                            auto tmp242 = [&]
                            {
                                auto tmp243 = c10::convert<int>(1L + (2L*x1));
                                auto tmp244 = static_cast<int>(0);
                                auto tmp245 = tmp243 >= tmp244;
                                auto tmp246 = static_cast<int>(56);
                                auto tmp247 = tmp243 < tmp246;
                                auto tmp248 = tmp245 & tmp247;
                                auto tmp249 = c10::convert<int>(1L + (2L*x2));
                                auto tmp250 = tmp249 >= tmp244;
                                auto tmp251 = tmp249 < tmp246;
                                auto tmp252 = tmp250 & tmp251;
                                auto tmp253 = tmp248 & tmp252;
                                auto tmp255 = tmp253 & tmp241;
                                auto tmp254 = [&]
                                {
                                    auto tmp256 = static_cast<float>(1.0);
                                    return tmp256;
                                }
                                ;
                                auto tmp257 = tmp253 ? tmp254() : static_cast<float>(1.0);
                                return tmp257;
                            }
                            ;
                            auto tmp258 = tmp241 ? tmp242() : static_cast<float>(0.0);
                            auto tmp259 = decltype(tmp258)(tmp258 + tmp240);
                            auto tmp260 = at::vec::Vectorized<float>(tmp259);
                            auto tmp261 = tmp69 / tmp260;
                            tmp261.store(out_ptr2 + static_cast<long>(x3 + (128L*x2) + (3584L*x1) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (28672L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x2 + (512L*x1) + (28672L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14336L + x2 + (512L*x1) + (28672L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14592L + x2 + (512L*x1) + (28672L*x0)));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (7168L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (401408L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (401408L*x0)));
                            auto tmp2 = tmp0 + tmp1;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(784.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__softmax_avg_pool2d_mul_sum_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                    auto tmp3 = at::vec::maximum(tmp1, tmp2);
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.exp();
                    tmp5.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (512L*x0)));
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (401408L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (401408L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (200704L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-1L) + (2L*x1));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(28);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<int>((-1L) + (2L*x2));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(out_ptr1 + static_cast<long>((-7424L) + x3 + (512L*x2) + (14336L*x1) + (200704L*x0)), to_float_mask(tmp10));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            auto tmp14 = c10::convert<int>(2L*x2);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = masked_load(out_ptr1 + static_cast<long>((-7168L) + x3 + (512L*x2) + (14336L*x1) + (200704L*x0)), to_float_mask(tmp18));
                                return tmp20;
                            }
                            ;
                            auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp18));
                            auto tmp22 = tmp21 + tmp13;
                            auto tmp23 = c10::convert<int>(1L + (2L*x2));
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = masked_load(out_ptr1 + static_cast<long>((-6912L) + x3 + (512L*x2) + (14336L*x1) + (200704L*x0)), to_float_mask(tmp27));
                                return tmp29;
                            }
                            ;
                            auto tmp30 = decltype(tmp28())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp28(), to_float_mask(tmp27));
                            auto tmp31 = tmp30 + tmp22;
                            auto tmp32 = c10::convert<int>(2L*x1);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = masked_load(out_ptr1 + static_cast<long>((-256L) + x3 + (512L*x2) + (14336L*x1) + (200704L*x0)), to_float_mask(tmp36));
                                return tmp38;
                            }
                            ;
                            auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp37(), to_float_mask(tmp36));
                            auto tmp40 = tmp39 + tmp31;
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(out_ptr1 + static_cast<long>(x3 + (512L*x2) + (14336L*x1) + (200704L*x0)), to_float_mask(tmp41));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp41));
                            auto tmp45 = tmp44 + tmp40;
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = masked_load(out_ptr1 + static_cast<long>(256L + x3 + (512L*x2) + (14336L*x1) + (200704L*x0)), to_float_mask(tmp46));
                                return tmp48;
                            }
                            ;
                            auto tmp49 = decltype(tmp47())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp47(), to_float_mask(tmp46));
                            auto tmp50 = tmp49 + tmp45;
                            auto tmp51 = c10::convert<int>(1L + (2L*x1));
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = masked_load(out_ptr1 + static_cast<long>(6912L + x3 + (512L*x2) + (14336L*x1) + (200704L*x0)), to_float_mask(tmp55));
                                return tmp57;
                            }
                            ;
                            auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp56(), to_float_mask(tmp55));
                            auto tmp59 = tmp58 + tmp50;
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = masked_load(out_ptr1 + static_cast<long>(7168L + x3 + (512L*x2) + (14336L*x1) + (200704L*x0)), to_float_mask(tmp60));
                                return tmp62;
                            }
                            ;
                            auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp61(), to_float_mask(tmp60));
                            auto tmp64 = tmp63 + tmp59;
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = masked_load(out_ptr1 + static_cast<long>(7424L + x3 + (512L*x2) + (14336L*x1) + (200704L*x0)), to_float_mask(tmp65));
                                return tmp67;
                            }
                            ;
                            auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp66(), to_float_mask(tmp65));
                            auto tmp69 = tmp68 + tmp64;
                            auto tmp70 = static_cast<int>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<int>(29);
                            auto tmp73 = tmp0 < tmp72;
                            auto tmp74 = tmp71 & tmp73;
                            auto tmp75 = tmp6 >= tmp70;
                            auto tmp76 = tmp6 < tmp72;
                            auto tmp77 = tmp75 & tmp76;
                            auto tmp78 = tmp74 & tmp77;
                            auto tmp79 = [&]
                            {
                                auto tmp80 = c10::convert<int>((-1L) + (2L*x1));
                                auto tmp81 = static_cast<int>(0);
                                auto tmp82 = tmp80 >= tmp81;
                                auto tmp83 = static_cast<int>(28);
                                auto tmp84 = tmp80 < tmp83;
                                auto tmp85 = tmp82 & tmp84;
                                auto tmp86 = c10::convert<int>((-1L) + (2L*x2));
                                auto tmp87 = tmp86 >= tmp81;
                                auto tmp88 = tmp86 < tmp83;
                                auto tmp89 = tmp87 & tmp88;
                                auto tmp90 = tmp85 & tmp89;
                                auto tmp92 = tmp90 & tmp78;
                                auto tmp91 = [&]
                                {
                                    auto tmp93 = static_cast<float>(1.0);
                                    return tmp93;
                                }
                                ;
                                auto tmp94 = tmp90 ? tmp91() : static_cast<float>(1.0);
                                return tmp94;
                            }
                            ;
                            auto tmp95 = tmp78 ? tmp79() : static_cast<float>(0.0);
                            auto tmp96 = tmp14 >= tmp70;
                            auto tmp97 = tmp14 < tmp72;
                            auto tmp98 = tmp96 & tmp97;
                            auto tmp99 = tmp74 & tmp98;
                            auto tmp100 = [&]
                            {
                                auto tmp101 = c10::convert<int>((-1L) + (2L*x1));
                                auto tmp102 = static_cast<int>(0);
                                auto tmp103 = tmp101 >= tmp102;
                                auto tmp104 = static_cast<int>(28);
                                auto tmp105 = tmp101 < tmp104;
                                auto tmp106 = tmp103 & tmp105;
                                auto tmp107 = c10::convert<int>(2L*x2);
                                auto tmp108 = tmp107 >= tmp102;
                                auto tmp109 = tmp107 < tmp104;
                                auto tmp110 = tmp108 & tmp109;
                                auto tmp111 = tmp106 & tmp110;
                                auto tmp113 = tmp111 & tmp99;
                                auto tmp112 = [&]
                                {
                                    auto tmp114 = static_cast<float>(1.0);
                                    return tmp114;
                                }
                                ;
                                auto tmp115 = tmp111 ? tmp112() : static_cast<float>(1.0);
                                return tmp115;
                            }
                            ;
                            auto tmp116 = tmp99 ? tmp100() : static_cast<float>(0.0);
                            auto tmp117 = decltype(tmp116)(tmp116 + tmp95);
                            auto tmp118 = tmp23 >= tmp70;
                            auto tmp119 = tmp23 < tmp72;
                            auto tmp120 = tmp118 & tmp119;
                            auto tmp121 = tmp74 & tmp120;
                            auto tmp122 = [&]
                            {
                                auto tmp123 = c10::convert<int>((-1L) + (2L*x1));
                                auto tmp124 = static_cast<int>(0);
                                auto tmp125 = tmp123 >= tmp124;
                                auto tmp126 = static_cast<int>(28);
                                auto tmp127 = tmp123 < tmp126;
                                auto tmp128 = tmp125 & tmp127;
                                auto tmp129 = c10::convert<int>(1L + (2L*x2));
                                auto tmp130 = tmp129 >= tmp124;
                                auto tmp131 = tmp129 < tmp126;
                                auto tmp132 = tmp130 & tmp131;
                                auto tmp133 = tmp128 & tmp132;
                                auto tmp135 = tmp133 & tmp121;
                                auto tmp134 = [&]
                                {
                                    auto tmp136 = static_cast<float>(1.0);
                                    return tmp136;
                                }
                                ;
                                auto tmp137 = tmp133 ? tmp134() : static_cast<float>(1.0);
                                return tmp137;
                            }
                            ;
                            auto tmp138 = tmp121 ? tmp122() : static_cast<float>(0.0);
                            auto tmp139 = decltype(tmp138)(tmp138 + tmp117);
                            auto tmp140 = tmp32 >= tmp70;
                            auto tmp141 = tmp32 < tmp72;
                            auto tmp142 = tmp140 & tmp141;
                            auto tmp143 = tmp142 & tmp77;
                            auto tmp144 = [&]
                            {
                                auto tmp145 = c10::convert<int>(2L*x1);
                                auto tmp146 = static_cast<int>(0);
                                auto tmp147 = tmp145 >= tmp146;
                                auto tmp148 = static_cast<int>(28);
                                auto tmp149 = tmp145 < tmp148;
                                auto tmp150 = tmp147 & tmp149;
                                auto tmp151 = c10::convert<int>((-1L) + (2L*x2));
                                auto tmp152 = tmp151 >= tmp146;
                                auto tmp153 = tmp151 < tmp148;
                                auto tmp154 = tmp152 & tmp153;
                                auto tmp155 = tmp150 & tmp154;
                                auto tmp157 = tmp155 & tmp143;
                                auto tmp156 = [&]
                                {
                                    auto tmp158 = static_cast<float>(1.0);
                                    return tmp158;
                                }
                                ;
                                auto tmp159 = tmp155 ? tmp156() : static_cast<float>(1.0);
                                return tmp159;
                            }
                            ;
                            auto tmp160 = tmp143 ? tmp144() : static_cast<float>(0.0);
                            auto tmp161 = decltype(tmp160)(tmp160 + tmp139);
                            auto tmp162 = tmp142 & tmp98;
                            auto tmp163 = [&]
                            {
                                auto tmp164 = c10::convert<int>(2L*x1);
                                auto tmp165 = static_cast<int>(0);
                                auto tmp166 = tmp164 >= tmp165;
                                auto tmp167 = static_cast<int>(28);
                                auto tmp168 = tmp164 < tmp167;
                                auto tmp169 = tmp166 & tmp168;
                                auto tmp170 = c10::convert<int>(2L*x2);
                                auto tmp171 = tmp170 >= tmp165;
                                auto tmp172 = tmp170 < tmp167;
                                auto tmp173 = tmp171 & tmp172;
                                auto tmp174 = tmp169 & tmp173;
                                auto tmp176 = tmp174 & tmp162;
                                auto tmp175 = [&]
                                {
                                    auto tmp177 = static_cast<float>(1.0);
                                    return tmp177;
                                }
                                ;
                                auto tmp178 = tmp174 ? tmp175() : static_cast<float>(1.0);
                                return tmp178;
                            }
                            ;
                            auto tmp179 = tmp162 ? tmp163() : static_cast<float>(0.0);
                            auto tmp180 = decltype(tmp179)(tmp179 + tmp161);
                            auto tmp181 = tmp142 & tmp120;
                            auto tmp182 = [&]
                            {
                                auto tmp183 = c10::convert<int>(2L*x1);
                                auto tmp184 = static_cast<int>(0);
                                auto tmp185 = tmp183 >= tmp184;
                                auto tmp186 = static_cast<int>(28);
                                auto tmp187 = tmp183 < tmp186;
                                auto tmp188 = tmp185 & tmp187;
                                auto tmp189 = c10::convert<int>(1L + (2L*x2));
                                auto tmp190 = tmp189 >= tmp184;
                                auto tmp191 = tmp189 < tmp186;
                                auto tmp192 = tmp190 & tmp191;
                                auto tmp193 = tmp188 & tmp192;
                                auto tmp195 = tmp193 & tmp181;
                                auto tmp194 = [&]
                                {
                                    auto tmp196 = static_cast<float>(1.0);
                                    return tmp196;
                                }
                                ;
                                auto tmp197 = tmp193 ? tmp194() : static_cast<float>(1.0);
                                return tmp197;
                            }
                            ;
                            auto tmp198 = tmp181 ? tmp182() : static_cast<float>(0.0);
                            auto tmp199 = decltype(tmp198)(tmp198 + tmp180);
                            auto tmp200 = tmp51 >= tmp70;
                            auto tmp201 = tmp51 < tmp72;
                            auto tmp202 = tmp200 & tmp201;
                            auto tmp203 = tmp202 & tmp77;
                            auto tmp204 = [&]
                            {
                                auto tmp205 = c10::convert<int>(1L + (2L*x1));
                                auto tmp206 = static_cast<int>(0);
                                auto tmp207 = tmp205 >= tmp206;
                                auto tmp208 = static_cast<int>(28);
                                auto tmp209 = tmp205 < tmp208;
                                auto tmp210 = tmp207 & tmp209;
                                auto tmp211 = c10::convert<int>((-1L) + (2L*x2));
                                auto tmp212 = tmp211 >= tmp206;
                                auto tmp213 = tmp211 < tmp208;
                                auto tmp214 = tmp212 & tmp213;
                                auto tmp215 = tmp210 & tmp214;
                                auto tmp217 = tmp215 & tmp203;
                                auto tmp216 = [&]
                                {
                                    auto tmp218 = static_cast<float>(1.0);
                                    return tmp218;
                                }
                                ;
                                auto tmp219 = tmp215 ? tmp216() : static_cast<float>(1.0);
                                return tmp219;
                            }
                            ;
                            auto tmp220 = tmp203 ? tmp204() : static_cast<float>(0.0);
                            auto tmp221 = decltype(tmp220)(tmp220 + tmp199);
                            auto tmp222 = tmp202 & tmp98;
                            auto tmp223 = [&]
                            {
                                auto tmp224 = c10::convert<int>(1L + (2L*x1));
                                auto tmp225 = static_cast<int>(0);
                                auto tmp226 = tmp224 >= tmp225;
                                auto tmp227 = static_cast<int>(28);
                                auto tmp228 = tmp224 < tmp227;
                                auto tmp229 = tmp226 & tmp228;
                                auto tmp230 = c10::convert<int>(2L*x2);
                                auto tmp231 = tmp230 >= tmp225;
                                auto tmp232 = tmp230 < tmp227;
                                auto tmp233 = tmp231 & tmp232;
                                auto tmp234 = tmp229 & tmp233;
                                auto tmp236 = tmp234 & tmp222;
                                auto tmp235 = [&]
                                {
                                    auto tmp237 = static_cast<float>(1.0);
                                    return tmp237;
                                }
                                ;
                                auto tmp238 = tmp234 ? tmp235() : static_cast<float>(1.0);
                                return tmp238;
                            }
                            ;
                            auto tmp239 = tmp222 ? tmp223() : static_cast<float>(0.0);
                            auto tmp240 = decltype(tmp239)(tmp239 + tmp221);
                            auto tmp241 = tmp202 & tmp120;
                            auto tmp242 = [&]
                            {
                                auto tmp243 = c10::convert<int>(1L + (2L*x1));
                                auto tmp244 = static_cast<int>(0);
                                auto tmp245 = tmp243 >= tmp244;
                                auto tmp246 = static_cast<int>(28);
                                auto tmp247 = tmp243 < tmp246;
                                auto tmp248 = tmp245 & tmp247;
                                auto tmp249 = c10::convert<int>(1L + (2L*x2));
                                auto tmp250 = tmp249 >= tmp244;
                                auto tmp251 = tmp249 < tmp246;
                                auto tmp252 = tmp250 & tmp251;
                                auto tmp253 = tmp248 & tmp252;
                                auto tmp255 = tmp253 & tmp241;
                                auto tmp254 = [&]
                                {
                                    auto tmp256 = static_cast<float>(1.0);
                                    return tmp256;
                                }
                                ;
                                auto tmp257 = tmp253 ? tmp254() : static_cast<float>(1.0);
                                return tmp257;
                            }
                            ;
                            auto tmp258 = tmp241 ? tmp242() : static_cast<float>(0.0);
                            auto tmp259 = decltype(tmp258)(tmp258 + tmp240);
                            auto tmp260 = at::vec::Vectorized<float>(tmp259);
                            auto tmp261 = tmp69 / tmp260;
                            tmp261.store(out_ptr2 + static_cast<long>(x3 + (256L*x2) + (3584L*x1) + (50176L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14336L + x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14848L + x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (7168L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x2) + (200704L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(512L + x1 + (1024L*x2) + (200704L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__softmax_avg_pool2d_mul_sum_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x2 + (1024L*x0)));
                    auto tmp3 = at::vec::maximum(tmp1, tmp2);
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.exp();
                    tmp5.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (1024L*x0)));
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1024L*x1) + (200704L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (1024L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(512L + x2 + (1024L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x2 + (1024L*x1) + (200704L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = c10::convert<int>((-1L) + (2L*x1));
                                auto tmp1 = static_cast<int>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<int>(14);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = tmp2 & tmp4;
                                auto tmp6 = c10::convert<int>((-1L) + (2L*x2));
                                auto tmp7 = tmp6 >= tmp1;
                                auto tmp8 = tmp6 < tmp3;
                                auto tmp9 = tmp7 & tmp8;
                                auto tmp10 = tmp5 & tmp9;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = masked_load(out_ptr1 + static_cast<long>((-7680L) + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0)), to_float_mask(tmp10));
                                    return tmp12;
                                }
                                ;
                                auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                                auto tmp14 = c10::convert<int>(2L*x2);
                                auto tmp15 = tmp14 >= tmp1;
                                auto tmp16 = tmp14 < tmp3;
                                auto tmp17 = tmp15 & tmp16;
                                auto tmp18 = tmp5 & tmp17;
                                auto tmp19 = [&]
                                {
                                    auto tmp20 = masked_load(out_ptr1 + static_cast<long>((-7168L) + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0)), to_float_mask(tmp18));
                                    return tmp20;
                                }
                                ;
                                auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp18));
                                auto tmp22 = tmp21 + tmp13;
                                auto tmp23 = c10::convert<int>(1L + (2L*x2));
                                auto tmp24 = tmp23 >= tmp1;
                                auto tmp25 = tmp23 < tmp3;
                                auto tmp26 = tmp24 & tmp25;
                                auto tmp27 = tmp5 & tmp26;
                                auto tmp28 = [&]
                                {
                                    auto tmp29 = masked_load(out_ptr1 + static_cast<long>((-6656L) + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0)), to_float_mask(tmp27));
                                    return tmp29;
                                }
                                ;
                                auto tmp30 = decltype(tmp28())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp28(), to_float_mask(tmp27));
                                auto tmp31 = tmp30 + tmp22;
                                auto tmp32 = c10::convert<int>(2L*x1);
                                auto tmp33 = tmp32 >= tmp1;
                                auto tmp34 = tmp32 < tmp3;
                                auto tmp35 = tmp33 & tmp34;
                                auto tmp36 = tmp35 & tmp9;
                                auto tmp37 = [&]
                                {
                                    auto tmp38 = masked_load(out_ptr1 + static_cast<long>((-512L) + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0)), to_float_mask(tmp36));
                                    return tmp38;
                                }
                                ;
                                auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp37(), to_float_mask(tmp36));
                                auto tmp40 = tmp39 + tmp31;
                                auto tmp41 = tmp35 & tmp17;
                                auto tmp42 = [&]
                                {
                                    auto tmp43 = masked_load(out_ptr1 + static_cast<long>(x3 + (1024L*x2) + (14336L*x1) + (100352L*x0)), to_float_mask(tmp41));
                                    return tmp43;
                                }
                                ;
                                auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp41));
                                auto tmp45 = tmp44 + tmp40;
                                auto tmp46 = tmp35 & tmp26;
                                auto tmp47 = [&]
                                {
                                    auto tmp48 = masked_load(out_ptr1 + static_cast<long>(512L + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0)), to_float_mask(tmp46));
                                    return tmp48;
                                }
                                ;
                                auto tmp49 = decltype(tmp47())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp47(), to_float_mask(tmp46));
                                auto tmp50 = tmp49 + tmp45;
                                auto tmp51 = c10::convert<int>(1L + (2L*x1));
                                auto tmp52 = tmp51 >= tmp1;
                                auto tmp53 = tmp51 < tmp3;
                                auto tmp54 = tmp52 & tmp53;
                                auto tmp55 = tmp54 & tmp9;
                                auto tmp56 = [&]
                                {
                                    auto tmp57 = masked_load(out_ptr1 + static_cast<long>(6656L + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0)), to_float_mask(tmp55));
                                    return tmp57;
                                }
                                ;
                                auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp56(), to_float_mask(tmp55));
                                auto tmp59 = tmp58 + tmp50;
                                auto tmp60 = tmp54 & tmp17;
                                auto tmp61 = [&]
                                {
                                    auto tmp62 = masked_load(out_ptr1 + static_cast<long>(7168L + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0)), to_float_mask(tmp60));
                                    return tmp62;
                                }
                                ;
                                auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp61(), to_float_mask(tmp60));
                                auto tmp64 = tmp63 + tmp59;
                                auto tmp65 = tmp54 & tmp26;
                                auto tmp66 = [&]
                                {
                                    auto tmp67 = masked_load(out_ptr1 + static_cast<long>(7680L + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0)), to_float_mask(tmp65));
                                    return tmp67;
                                }
                                ;
                                auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp66(), to_float_mask(tmp65));
                                auto tmp69 = tmp68 + tmp64;
                                auto tmp70 = static_cast<int>(-1);
                                auto tmp71 = tmp0 >= tmp70;
                                auto tmp72 = static_cast<int>(15);
                                auto tmp73 = tmp0 < tmp72;
                                auto tmp74 = tmp71 & tmp73;
                                auto tmp75 = tmp6 >= tmp70;
                                auto tmp76 = tmp6 < tmp72;
                                auto tmp77 = tmp75 & tmp76;
                                auto tmp78 = tmp74 & tmp77;
                                auto tmp79 = [&]
                                {
                                    auto tmp80 = c10::convert<int>((-1L) + (2L*x1));
                                    auto tmp81 = static_cast<int>(0);
                                    auto tmp82 = tmp80 >= tmp81;
                                    auto tmp83 = static_cast<int>(14);
                                    auto tmp84 = tmp80 < tmp83;
                                    auto tmp85 = tmp82 & tmp84;
                                    auto tmp86 = c10::convert<int>((-1L) + (2L*x2));
                                    auto tmp87 = tmp86 >= tmp81;
                                    auto tmp88 = tmp86 < tmp83;
                                    auto tmp89 = tmp87 & tmp88;
                                    auto tmp90 = tmp85 & tmp89;
                                    auto tmp92 = tmp90 & tmp78;
                                    auto tmp91 = [&]
                                    {
                                        auto tmp93 = static_cast<float>(1.0);
                                        return tmp93;
                                    }
                                    ;
                                    auto tmp94 = tmp90 ? tmp91() : static_cast<float>(1.0);
                                    return tmp94;
                                }
                                ;
                                auto tmp95 = tmp78 ? tmp79() : static_cast<float>(0.0);
                                auto tmp96 = tmp14 >= tmp70;
                                auto tmp97 = tmp14 < tmp72;
                                auto tmp98 = tmp96 & tmp97;
                                auto tmp99 = tmp74 & tmp98;
                                auto tmp100 = [&]
                                {
                                    auto tmp101 = c10::convert<int>((-1L) + (2L*x1));
                                    auto tmp102 = static_cast<int>(0);
                                    auto tmp103 = tmp101 >= tmp102;
                                    auto tmp104 = static_cast<int>(14);
                                    auto tmp105 = tmp101 < tmp104;
                                    auto tmp106 = tmp103 & tmp105;
                                    auto tmp107 = c10::convert<int>(2L*x2);
                                    auto tmp108 = tmp107 >= tmp102;
                                    auto tmp109 = tmp107 < tmp104;
                                    auto tmp110 = tmp108 & tmp109;
                                    auto tmp111 = tmp106 & tmp110;
                                    auto tmp113 = tmp111 & tmp99;
                                    auto tmp112 = [&]
                                    {
                                        auto tmp114 = static_cast<float>(1.0);
                                        return tmp114;
                                    }
                                    ;
                                    auto tmp115 = tmp111 ? tmp112() : static_cast<float>(1.0);
                                    return tmp115;
                                }
                                ;
                                auto tmp116 = tmp99 ? tmp100() : static_cast<float>(0.0);
                                auto tmp117 = decltype(tmp116)(tmp116 + tmp95);
                                auto tmp118 = tmp23 >= tmp70;
                                auto tmp119 = tmp23 < tmp72;
                                auto tmp120 = tmp118 & tmp119;
                                auto tmp121 = tmp74 & tmp120;
                                auto tmp122 = [&]
                                {
                                    auto tmp123 = c10::convert<int>((-1L) + (2L*x1));
                                    auto tmp124 = static_cast<int>(0);
                                    auto tmp125 = tmp123 >= tmp124;
                                    auto tmp126 = static_cast<int>(14);
                                    auto tmp127 = tmp123 < tmp126;
                                    auto tmp128 = tmp125 & tmp127;
                                    auto tmp129 = c10::convert<int>(1L + (2L*x2));
                                    auto tmp130 = tmp129 >= tmp124;
                                    auto tmp131 = tmp129 < tmp126;
                                    auto tmp132 = tmp130 & tmp131;
                                    auto tmp133 = tmp128 & tmp132;
                                    auto tmp135 = tmp133 & tmp121;
                                    auto tmp134 = [&]
                                    {
                                        auto tmp136 = static_cast<float>(1.0);
                                        return tmp136;
                                    }
                                    ;
                                    auto tmp137 = tmp133 ? tmp134() : static_cast<float>(1.0);
                                    return tmp137;
                                }
                                ;
                                auto tmp138 = tmp121 ? tmp122() : static_cast<float>(0.0);
                                auto tmp139 = decltype(tmp138)(tmp138 + tmp117);
                                auto tmp140 = tmp32 >= tmp70;
                                auto tmp141 = tmp32 < tmp72;
                                auto tmp142 = tmp140 & tmp141;
                                auto tmp143 = tmp142 & tmp77;
                                auto tmp144 = [&]
                                {
                                    auto tmp145 = c10::convert<int>(2L*x1);
                                    auto tmp146 = static_cast<int>(0);
                                    auto tmp147 = tmp145 >= tmp146;
                                    auto tmp148 = static_cast<int>(14);
                                    auto tmp149 = tmp145 < tmp148;
                                    auto tmp150 = tmp147 & tmp149;
                                    auto tmp151 = c10::convert<int>((-1L) + (2L*x2));
                                    auto tmp152 = tmp151 >= tmp146;
                                    auto tmp153 = tmp151 < tmp148;
                                    auto tmp154 = tmp152 & tmp153;
                                    auto tmp155 = tmp150 & tmp154;
                                    auto tmp157 = tmp155 & tmp143;
                                    auto tmp156 = [&]
                                    {
                                        auto tmp158 = static_cast<float>(1.0);
                                        return tmp158;
                                    }
                                    ;
                                    auto tmp159 = tmp155 ? tmp156() : static_cast<float>(1.0);
                                    return tmp159;
                                }
                                ;
                                auto tmp160 = tmp143 ? tmp144() : static_cast<float>(0.0);
                                auto tmp161 = decltype(tmp160)(tmp160 + tmp139);
                                auto tmp162 = tmp142 & tmp98;
                                auto tmp163 = [&]
                                {
                                    auto tmp164 = c10::convert<int>(2L*x1);
                                    auto tmp165 = static_cast<int>(0);
                                    auto tmp166 = tmp164 >= tmp165;
                                    auto tmp167 = static_cast<int>(14);
                                    auto tmp168 = tmp164 < tmp167;
                                    auto tmp169 = tmp166 & tmp168;
                                    auto tmp170 = c10::convert<int>(2L*x2);
                                    auto tmp171 = tmp170 >= tmp165;
                                    auto tmp172 = tmp170 < tmp167;
                                    auto tmp173 = tmp171 & tmp172;
                                    auto tmp174 = tmp169 & tmp173;
                                    auto tmp176 = tmp174 & tmp162;
                                    auto tmp175 = [&]
                                    {
                                        auto tmp177 = static_cast<float>(1.0);
                                        return tmp177;
                                    }
                                    ;
                                    auto tmp178 = tmp174 ? tmp175() : static_cast<float>(1.0);
                                    return tmp178;
                                }
                                ;
                                auto tmp179 = tmp162 ? tmp163() : static_cast<float>(0.0);
                                auto tmp180 = decltype(tmp179)(tmp179 + tmp161);
                                auto tmp181 = tmp142 & tmp120;
                                auto tmp182 = [&]
                                {
                                    auto tmp183 = c10::convert<int>(2L*x1);
                                    auto tmp184 = static_cast<int>(0);
                                    auto tmp185 = tmp183 >= tmp184;
                                    auto tmp186 = static_cast<int>(14);
                                    auto tmp187 = tmp183 < tmp186;
                                    auto tmp188 = tmp185 & tmp187;
                                    auto tmp189 = c10::convert<int>(1L + (2L*x2));
                                    auto tmp190 = tmp189 >= tmp184;
                                    auto tmp191 = tmp189 < tmp186;
                                    auto tmp192 = tmp190 & tmp191;
                                    auto tmp193 = tmp188 & tmp192;
                                    auto tmp195 = tmp193 & tmp181;
                                    auto tmp194 = [&]
                                    {
                                        auto tmp196 = static_cast<float>(1.0);
                                        return tmp196;
                                    }
                                    ;
                                    auto tmp197 = tmp193 ? tmp194() : static_cast<float>(1.0);
                                    return tmp197;
                                }
                                ;
                                auto tmp198 = tmp181 ? tmp182() : static_cast<float>(0.0);
                                auto tmp199 = decltype(tmp198)(tmp198 + tmp180);
                                auto tmp200 = tmp51 >= tmp70;
                                auto tmp201 = tmp51 < tmp72;
                                auto tmp202 = tmp200 & tmp201;
                                auto tmp203 = tmp202 & tmp77;
                                auto tmp204 = [&]
                                {
                                    auto tmp205 = c10::convert<int>(1L + (2L*x1));
                                    auto tmp206 = static_cast<int>(0);
                                    auto tmp207 = tmp205 >= tmp206;
                                    auto tmp208 = static_cast<int>(14);
                                    auto tmp209 = tmp205 < tmp208;
                                    auto tmp210 = tmp207 & tmp209;
                                    auto tmp211 = c10::convert<int>((-1L) + (2L*x2));
                                    auto tmp212 = tmp211 >= tmp206;
                                    auto tmp213 = tmp211 < tmp208;
                                    auto tmp214 = tmp212 & tmp213;
                                    auto tmp215 = tmp210 & tmp214;
                                    auto tmp217 = tmp215 & tmp203;
                                    auto tmp216 = [&]
                                    {
                                        auto tmp218 = static_cast<float>(1.0);
                                        return tmp218;
                                    }
                                    ;
                                    auto tmp219 = tmp215 ? tmp216() : static_cast<float>(1.0);
                                    return tmp219;
                                }
                                ;
                                auto tmp220 = tmp203 ? tmp204() : static_cast<float>(0.0);
                                auto tmp221 = decltype(tmp220)(tmp220 + tmp199);
                                auto tmp222 = tmp202 & tmp98;
                                auto tmp223 = [&]
                                {
                                    auto tmp224 = c10::convert<int>(1L + (2L*x1));
                                    auto tmp225 = static_cast<int>(0);
                                    auto tmp226 = tmp224 >= tmp225;
                                    auto tmp227 = static_cast<int>(14);
                                    auto tmp228 = tmp224 < tmp227;
                                    auto tmp229 = tmp226 & tmp228;
                                    auto tmp230 = c10::convert<int>(2L*x2);
                                    auto tmp231 = tmp230 >= tmp225;
                                    auto tmp232 = tmp230 < tmp227;
                                    auto tmp233 = tmp231 & tmp232;
                                    auto tmp234 = tmp229 & tmp233;
                                    auto tmp236 = tmp234 & tmp222;
                                    auto tmp235 = [&]
                                    {
                                        auto tmp237 = static_cast<float>(1.0);
                                        return tmp237;
                                    }
                                    ;
                                    auto tmp238 = tmp234 ? tmp235() : static_cast<float>(1.0);
                                    return tmp238;
                                }
                                ;
                                auto tmp239 = tmp222 ? tmp223() : static_cast<float>(0.0);
                                auto tmp240 = decltype(tmp239)(tmp239 + tmp221);
                                auto tmp241 = tmp202 & tmp120;
                                auto tmp242 = [&]
                                {
                                    auto tmp243 = c10::convert<int>(1L + (2L*x1));
                                    auto tmp244 = static_cast<int>(0);
                                    auto tmp245 = tmp243 >= tmp244;
                                    auto tmp246 = static_cast<int>(14);
                                    auto tmp247 = tmp243 < tmp246;
                                    auto tmp248 = tmp245 & tmp247;
                                    auto tmp249 = c10::convert<int>(1L + (2L*x2));
                                    auto tmp250 = tmp249 >= tmp244;
                                    auto tmp251 = tmp249 < tmp246;
                                    auto tmp252 = tmp250 & tmp251;
                                    auto tmp253 = tmp248 & tmp252;
                                    auto tmp255 = tmp253 & tmp241;
                                    auto tmp254 = [&]
                                    {
                                        auto tmp256 = static_cast<float>(1.0);
                                        return tmp256;
                                    }
                                    ;
                                    auto tmp257 = tmp253 ? tmp254() : static_cast<float>(1.0);
                                    return tmp257;
                                }
                                ;
                                auto tmp258 = tmp241 ? tmp242() : static_cast<float>(0.0);
                                auto tmp259 = decltype(tmp258)(tmp258 + tmp240);
                                auto tmp260 = at::vec::Vectorized<float>(tmp259);
                                auto tmp261 = tmp69 / tmp260;
                                tmp261.store(out_ptr2 + static_cast<long>(x3 + (512L*x2) + (3584L*x1) + (25088L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(28L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (2048L*x1) + (28672L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + x2 + (2048L*x1) + (28672L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14336L + x2 + (2048L*x1) + (28672L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(15360L + x2 + (2048L*x1) + (28672L*x0)));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (7168L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x2) + (100352L*x0)));
                            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, ), (1, ))
    assert_size_stride(arg12_1, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg16_1, (32, ), (1, ))
    assert_size_stride(arg17_1, (32, ), (1, ))
    assert_size_stride(arg18_1, (32, ), (1, ))
    assert_size_stride(arg19_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg25_1, (256, ), (1, ))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg28_1, (128, ), (1, ))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg34_1, (64, ), (1, ))
    assert_size_stride(arg35_1, (64, ), (1, ))
    assert_size_stride(arg36_1, (64, ), (1, ))
    assert_size_stride(arg37_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg40_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg43_1, (512, ), (1, ))
    assert_size_stride(arg44_1, (512, ), (1, ))
    assert_size_stride(arg45_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg49_1, (512, ), (1, ))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg52_1, (128, ), (1, ))
    assert_size_stride(arg53_1, (128, ), (1, ))
    assert_size_stride(arg54_1, (128, ), (1, ))
    assert_size_stride(arg55_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg56_1, (512, ), (1, ))
    assert_size_stride(arg57_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg58_1, (1024, ), (1, ))
    assert_size_stride(arg59_1, (1024, ), (1, ))
    assert_size_stride(arg60_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg61_1, (1024, ), (1, ))
    assert_size_stride(arg62_1, (1024, ), (1, ))
    assert_size_stride(arg63_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg64_1, (512, ), (1, ))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg67_1, (1024, ), (1, ))
    assert_size_stride(arg68_1, (1024, ), (1, ))
    assert_size_stride(arg69_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg70_1, (256, ), (1, ))
    assert_size_stride(arg71_1, (256, ), (1, ))
    assert_size_stride(arg72_1, (256, ), (1, ))
    assert_size_stride(arg73_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg74_1, (1024, ), (1, ))
    assert_size_stride(arg75_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg76_1, (2048, ), (1, ))
    assert_size_stride(arg77_1, (2048, ), (1, ))
    assert_size_stride(arg78_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg79_1, (2048, ), (1, ))
    assert_size_stride(arg80_1, (2048, ), (1, ))
    assert_size_stride(arg81_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg82_1, (1000, ), (1, ))
    assert_size_stride(arg83_1, (32, ), (1, ))
    assert_size_stride(arg84_1, (32, ), (1, ))
    assert_size_stride(arg85_1, (), ())
    assert_size_stride(arg86_1, (32, ), (1, ))
    assert_size_stride(arg87_1, (32, ), (1, ))
    assert_size_stride(arg88_1, (), ())
    assert_size_stride(arg89_1, (64, ), (1, ))
    assert_size_stride(arg90_1, (64, ), (1, ))
    assert_size_stride(arg91_1, (), ())
    assert_size_stride(arg92_1, (64, ), (1, ))
    assert_size_stride(arg93_1, (64, ), (1, ))
    assert_size_stride(arg94_1, (), ())
    assert_size_stride(arg95_1, (128, ), (1, ))
    assert_size_stride(arg96_1, (128, ), (1, ))
    assert_size_stride(arg97_1, (), ())
    assert_size_stride(arg98_1, (32, ), (1, ))
    assert_size_stride(arg99_1, (32, ), (1, ))
    assert_size_stride(arg100_1, (), ())
    assert_size_stride(arg101_1, (256, ), (1, ))
    assert_size_stride(arg102_1, (256, ), (1, ))
    assert_size_stride(arg103_1, (), ())
    assert_size_stride(arg104_1, (256, ), (1, ))
    assert_size_stride(arg105_1, (256, ), (1, ))
    assert_size_stride(arg106_1, (), ())
    assert_size_stride(arg107_1, (128, ), (1, ))
    assert_size_stride(arg108_1, (128, ), (1, ))
    assert_size_stride(arg109_1, (), ())
    assert_size_stride(arg110_1, (256, ), (1, ))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (), ())
    assert_size_stride(arg113_1, (64, ), (1, ))
    assert_size_stride(arg114_1, (64, ), (1, ))
    assert_size_stride(arg115_1, (), ())
    assert_size_stride(arg116_1, (512, ), (1, ))
    assert_size_stride(arg117_1, (512, ), (1, ))
    assert_size_stride(arg118_1, (), ())
    assert_size_stride(arg119_1, (512, ), (1, ))
    assert_size_stride(arg120_1, (512, ), (1, ))
    assert_size_stride(arg121_1, (), ())
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (256, ), (1, ))
    assert_size_stride(arg124_1, (), ())
    assert_size_stride(arg125_1, (512, ), (1, ))
    assert_size_stride(arg126_1, (512, ), (1, ))
    assert_size_stride(arg127_1, (), ())
    assert_size_stride(arg128_1, (128, ), (1, ))
    assert_size_stride(arg129_1, (128, ), (1, ))
    assert_size_stride(arg130_1, (), ())
    assert_size_stride(arg131_1, (1024, ), (1, ))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (), ())
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (1024, ), (1, ))
    assert_size_stride(arg136_1, (), ())
    assert_size_stride(arg137_1, (512, ), (1, ))
    assert_size_stride(arg138_1, (512, ), (1, ))
    assert_size_stride(arg139_1, (), ())
    assert_size_stride(arg140_1, (1024, ), (1, ))
    assert_size_stride(arg141_1, (1024, ), (1, ))
    assert_size_stride(arg142_1, (), ())
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (), ())
    assert_size_stride(arg146_1, (2048, ), (1, ))
    assert_size_stride(arg147_1, (2048, ), (1, ))
    assert_size_stride(arg148_1, (), ())
    assert_size_stride(arg149_1, (2048, ), (1, ))
    assert_size_stride(arg150_1, (2048, ), (1, ))
    assert_size_stride(arg151_1, (), ())
    assert_size_stride(arg152_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg152_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg152_1
    # Source Nodes: [l__mod___conv1_0], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (4, 32, 112, 112), (401408, 1, 3584, 32))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = empty_strided((32, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg1_1
    del arg2_1
    del arg3_1
    del arg83_1
    del arg84_1
    # Source Nodes: [l__mod___conv1_1, l__mod___conv1_2, l__mod___conv1_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf5, (4, 32, 112, 112), (401408, 1, 3584, 32))
    del buf3
    del buf4
    buf6 = buf5; del buf5  # reuse
    buf7 = empty_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_2(c_void_p(buf6.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf7.data_ptr()))
    del arg4_1
    del arg5_1
    del arg6_1
    del arg86_1
    del arg87_1
    # Source Nodes: [l__mod___conv1_4, l__mod___conv1_5, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf8 = extern_kernels.convolution(buf6, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf8, (4, 64, 112, 112), (802816, 1, 7168, 64))
    del buf6
    del buf7
    buf9 = buf8; del buf8  # reuse
    buf10 = empty_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3(c_void_p(buf9.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf10.data_ptr()))
    del arg7_1
    del arg89_1
    del arg8_1
    del arg90_1
    del buf9
    # Source Nodes: [out], Original ATen: [aten.convolution]
    buf11 = extern_kernels.convolution(buf10, arg9_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf11, (4, 64, 56, 56), (200704, 1, 3584, 64))
    del arg9_1
    buf12 = buf11; del buf11  # reuse
    buf13 = empty_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_4(c_void_p(buf12.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(buf13.data_ptr()))
    del arg10_1
    del arg11_1
    del arg12_1
    del arg92_1
    del arg93_1
    # Source Nodes: [out_1, out_2, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf14 = extern_kernels.convolution(buf12, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf14, (4, 128, 56, 56), (401408, 1, 7168, 128))
    del buf13
    buf15 = buf14; del buf14  # reuse
    buf16 = empty_strided((4, 64, 1, 1), (64, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf17 = reinterpret_tensor(buf16, (4, 64, 1, 1), (64, 1, 64, 64), 0); del buf16  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_5(c_void_p(buf15.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg14_1.data_ptr()))
    del arg13_1
    del arg14_1
    del arg95_1
    del arg96_1
    # Source Nodes: [x_gap, x_gap_1, x_gap_2], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf18 = extern_kernels.convolution(buf17, arg15_1, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf18, (4, 32, 1, 1), (32, 1, 32, 32))
    del arg15_1
    del arg16_1
    del buf17
    buf19 = buf18; del buf18  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_6(c_void_p(buf19.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(arg18_1.data_ptr()))
    del arg17_1
    del arg18_1
    del arg98_1
    del arg99_1
    # Source Nodes: [x_attn, x_gap_3, x_gap_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf20 = extern_kernels.convolution(buf19, arg19_1, arg20_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf20, (4, 128, 1, 1), (128, 1, 128, 128))
    del arg19_1
    del arg20_1
    del buf19
    buf21 = empty_strided((4, 2, 1, 64), (128, 64, 512, 1), device='cpu', dtype=torch.float32)
    buf22 = buf12; del buf12  # reuse
    cpp_fused__softmax_mul_sum_7(c_void_p(buf20.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    del buf15
    del buf20
    # Source Nodes: [mul, out_3, out_8], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf23 = extern_kernels.convolution(buf22, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf23, (4, 256, 56, 56), (802816, 1, 14336, 256))
    del arg21_1
    del buf22
    # Source Nodes: [getattr_l__mod___layer1___0___downsample_1], Original ATen: [aten.convolution]
    buf24 = extern_kernels.convolution(buf10, arg24_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf24, (4, 256, 56, 56), (802816, 1, 14336, 256))
    del arg24_1
    buf25 = buf23; del buf23  # reuse
    buf26 = buf25; del buf25  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_8(c_void_p(buf26.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(arg26_1.data_ptr()))
    del arg101_1
    del arg102_1
    del arg104_1
    del arg105_1
    del arg22_1
    del arg23_1
    del arg25_1
    del arg26_1
    del buf24
    # Source Nodes: [out_12, shortcut_2], Original ATen: [aten.convolution, aten.relu]
    buf27 = extern_kernels.convolution(buf26, arg27_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf27, (4, 128, 56, 56), (401408, 1, 7168, 128))
    del arg27_1
    buf28 = buf27; del buf27  # reuse
    buf29 = empty_strided((256, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_9(c_void_p(buf28.data_ptr()), c_void_p(arg107_1.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(buf29.data_ptr()))
    del arg107_1
    del arg108_1
    del arg28_1
    del arg29_1
    del arg30_1
    # Source Nodes: [out_13, out_14, x_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf30 = extern_kernels.convolution(buf28, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf30, (4, 256, 56, 56), (802816, 1, 14336, 256))
    del buf29
    buf31 = buf30; del buf30  # reuse
    buf32 = reinterpret_tensor(buf21, (4, 128, 1, 1), (128, 1, 512, 512), 0); del buf21  # reuse
    buf33 = reinterpret_tensor(buf32, (4, 128, 1, 1), (128, 1, 128, 128), 0); del buf32  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_10(c_void_p(buf31.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg32_1.data_ptr()))
    del arg110_1
    del arg111_1
    del arg31_1
    del arg32_1
    # Source Nodes: [x_gap_5, x_gap_6, x_gap_7], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf34 = extern_kernels.convolution(buf33, arg33_1, arg34_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf34, (4, 64, 1, 1), (64, 1, 64, 64))
    del arg33_1
    del arg34_1
    del buf33
    buf35 = buf34; del buf34  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_11(c_void_p(buf35.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(arg36_1.data_ptr()))
    del arg113_1
    del arg114_1
    del arg35_1
    del arg36_1
    # Source Nodes: [x_attn_2, x_gap_8, x_gap_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf36 = extern_kernels.convolution(buf35, arg37_1, arg38_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf36, (4, 256, 1, 1), (256, 1, 256, 256))
    del arg37_1
    del arg38_1
    del buf35
    buf37 = empty_strided((4, 2, 1, 128), (256, 128, 1024, 1), device='cpu', dtype=torch.float32)
    buf38 = buf28; del buf28  # reuse
    buf39 = empty_strided((4, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_avg_pool2d_mul_sum_12(c_void_p(buf36.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()))
    del buf31
    del buf36
    del buf38
    # Source Nodes: [out_21], Original ATen: [aten.convolution]
    buf40 = extern_kernels.convolution(buf39, arg39_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf40, (4, 512, 28, 28), (401408, 1, 14336, 512))
    del arg39_1
    buf41 = reinterpret_tensor(buf10, (4, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf10  # reuse
    cpp_fused_avg_pool2d_13(c_void_p(buf26.data_ptr()), c_void_p(buf41.data_ptr()))
    del buf26
    # Source Nodes: [getattr_l__mod___layer2___0___downsample_0, getattr_l__mod___layer2___0___downsample_1], Original ATen: [aten.avg_pool2d, aten.convolution]
    buf42 = extern_kernels.convolution(buf41, arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf42, (4, 512, 28, 28), (401408, 1, 14336, 512))
    del arg42_1
    del buf41
    buf43 = buf40; del buf40  # reuse
    buf44 = buf43; del buf43  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_14(c_void_p(buf44.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(arg44_1.data_ptr()))
    del arg116_1
    del arg117_1
    del arg119_1
    del arg120_1
    del arg40_1
    del arg41_1
    del arg43_1
    del arg44_1
    del buf42
    # Source Nodes: [out_25, shortcut_4], Original ATen: [aten.convolution, aten.relu]
    buf45 = extern_kernels.convolution(buf44, arg45_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf45, (4, 256, 28, 28), (200704, 1, 7168, 256))
    del arg45_1
    buf46 = buf45; del buf45  # reuse
    buf47 = empty_strided((512, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_15(c_void_p(buf46.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(buf47.data_ptr()))
    del arg122_1
    del arg123_1
    del arg46_1
    del arg47_1
    del arg48_1
    # Source Nodes: [out_26, out_27, x_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf48 = extern_kernels.convolution(buf46, buf47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf48, (4, 512, 28, 28), (401408, 1, 14336, 512))
    del buf47
    buf49 = buf48; del buf48  # reuse
    buf50 = reinterpret_tensor(buf37, (4, 256, 1, 1), (256, 1, 1024, 1024), 0); del buf37  # reuse
    buf51 = reinterpret_tensor(buf50, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf50  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_16(c_void_p(buf49.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(arg50_1.data_ptr()))
    del arg125_1
    del arg126_1
    del arg49_1
    del arg50_1
    # Source Nodes: [x_gap_10, x_gap_11, x_gap_12], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf52 = extern_kernels.convolution(buf51, arg51_1, arg52_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf52, (4, 128, 1, 1), (128, 1, 128, 128))
    del arg51_1
    del arg52_1
    del buf51
    buf53 = buf52; del buf52  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_17(c_void_p(buf53.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(arg129_1.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(arg54_1.data_ptr()))
    del arg128_1
    del arg129_1
    del arg53_1
    del arg54_1
    # Source Nodes: [x_attn_4, x_gap_13, x_gap_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf54 = extern_kernels.convolution(buf53, arg55_1, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf54, (4, 512, 1, 1), (512, 1, 512, 512))
    del arg55_1
    del arg56_1
    del buf53
    buf55 = empty_strided((4, 2, 1, 256), (512, 256, 2048, 1), device='cpu', dtype=torch.float32)
    buf56 = buf46; del buf46  # reuse
    buf57 = empty_strided((4, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_avg_pool2d_mul_sum_18(c_void_p(buf54.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()))
    del buf49
    del buf54
    del buf56
    # Source Nodes: [out_34], Original ATen: [aten.convolution]
    buf58 = extern_kernels.convolution(buf57, arg57_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf58, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg57_1
    buf59 = reinterpret_tensor(buf39, (4, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf39  # reuse
    cpp_fused_avg_pool2d_19(c_void_p(buf44.data_ptr()), c_void_p(buf59.data_ptr()))
    del buf44
    # Source Nodes: [getattr_l__mod___layer3___0___downsample_0, getattr_l__mod___layer3___0___downsample_1], Original ATen: [aten.avg_pool2d, aten.convolution]
    buf60 = extern_kernels.convolution(buf59, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf60, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg60_1
    del buf59
    buf61 = buf58; del buf58  # reuse
    buf62 = buf61; del buf61  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_20(c_void_p(buf62.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(arg135_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg62_1.data_ptr()))
    del arg131_1
    del arg132_1
    del arg134_1
    del arg135_1
    del arg58_1
    del arg59_1
    del arg61_1
    del arg62_1
    del buf60
    # Source Nodes: [out_38, shortcut_6], Original ATen: [aten.convolution, aten.relu]
    buf63 = extern_kernels.convolution(buf62, arg63_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf63, (4, 512, 14, 14), (100352, 1, 7168, 512))
    del arg63_1
    buf64 = buf63; del buf63  # reuse
    buf65 = empty_strided((1024, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_21(c_void_p(buf64.data_ptr()), c_void_p(arg137_1.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(buf65.data_ptr()))
    del arg137_1
    del arg138_1
    del arg64_1
    del arg65_1
    del arg66_1
    # Source Nodes: [out_39, out_40, x_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf66 = extern_kernels.convolution(buf64, buf65, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf66, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    del buf65
    buf67 = buf66; del buf66  # reuse
    buf68 = reinterpret_tensor(buf55, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf55  # reuse
    buf69 = reinterpret_tensor(buf68, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf68  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_22(c_void_p(buf67.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()))
    del arg140_1
    del arg141_1
    del arg67_1
    del arg68_1
    # Source Nodes: [x_gap_15, x_gap_16, x_gap_17], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf70 = extern_kernels.convolution(buf69, arg69_1, arg70_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf70, (4, 256, 1, 1), (256, 1, 256, 256))
    del arg69_1
    del arg70_1
    del buf69
    buf71 = buf70; del buf70  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_23(c_void_p(buf71.data_ptr()), c_void_p(arg143_1.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(arg71_1.data_ptr()), c_void_p(arg72_1.data_ptr()))
    del arg143_1
    del arg144_1
    del arg71_1
    del arg72_1
    # Source Nodes: [x_attn_6, x_gap_18, x_gap_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf72 = extern_kernels.convolution(buf71, arg73_1, arg74_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf72, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
    del arg73_1
    del arg74_1
    del buf71
    buf73 = empty_strided((4, 2, 1, 512), (1024, 512, 4096, 1), device='cpu', dtype=torch.float32)
    buf74 = buf64; del buf64  # reuse
    buf75 = empty_strided((4, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_avg_pool2d_mul_sum_24(c_void_p(buf72.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()))
    del buf67
    del buf72
    del buf73
    del buf74
    # Source Nodes: [out_47], Original ATen: [aten.convolution]
    buf76 = extern_kernels.convolution(buf75, arg75_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf76, (4, 2048, 7, 7), (100352, 1, 14336, 2048))
    del arg75_1
    del buf75
    buf77 = reinterpret_tensor(buf57, (4, 1024, 7, 7), (50176, 1, 7168, 1024), 0); del buf57  # reuse
    cpp_fused_avg_pool2d_25(c_void_p(buf62.data_ptr()), c_void_p(buf77.data_ptr()))
    del buf62
    # Source Nodes: [getattr_l__mod___layer4___0___downsample_0, getattr_l__mod___layer4___0___downsample_1], Original ATen: [aten.avg_pool2d, aten.convolution]
    buf78 = extern_kernels.convolution(buf77, arg78_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf78, (4, 2048, 7, 7), (100352, 1, 14336, 2048))
    del arg78_1
    del buf77
    buf79 = buf76; del buf76  # reuse
    buf80 = empty_strided((4, 2048, 1, 1), (2048, 1, 8192, 8192), device='cpu', dtype=torch.float32)
    buf81 = reinterpret_tensor(buf80, (4, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf80  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_26(c_void_p(buf79.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(arg147_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(arg149_1.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(arg80_1.data_ptr()))
    del arg146_1
    del arg147_1
    del arg149_1
    del arg150_1
    del arg76_1
    del arg77_1
    del arg79_1
    del arg80_1
    del buf78
    del buf79
    buf82 = empty((4, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_44], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg82_1, reinterpret_tensor(buf81, (4, 2048), (2048, 1), 0), reinterpret_tensor(arg81_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf82)
    del arg81_1
    del arg82_1
    return (buf82, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg86_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg89_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg92_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg95_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg98_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg101_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg104_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg107_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg110_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg113_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg116_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg119_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg122_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg125_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg128_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg131_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg134_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg137_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg140_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg143_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg146_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg149_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg152_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('timm_resnest', benchmark_compiled_module)
