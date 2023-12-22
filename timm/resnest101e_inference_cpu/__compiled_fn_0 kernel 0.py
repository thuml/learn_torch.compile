
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
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
                                auto tmp12 = masked_load(in_out_ptr0 + static_cast<long>((-16512L) + x3 + (256L*x2) + (32768L*x1) + (2097152L*x0)), to_float_mask(tmp10));
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
                                auto tmp20 = masked_load(in_out_ptr0 + static_cast<long>((-16384L) + x3 + (256L*x2) + (32768L*x1) + (2097152L*x0)), to_float_mask(tmp18));
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
                                auto tmp29 = masked_load(in_out_ptr0 + static_cast<long>((-16256L) + x3 + (256L*x2) + (32768L*x1) + (2097152L*x0)), to_float_mask(tmp27));
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
                                auto tmp38 = masked_load(in_out_ptr0 + static_cast<long>((-128L) + x3 + (256L*x2) + (32768L*x1) + (2097152L*x0)), to_float_mask(tmp36));
                                return tmp38;
                            }
                            ;
                            auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp37(), to_float_mask(tmp36));
                            auto tmp40 = at::vec::maximum(tmp39, tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(in_out_ptr0 + static_cast<long>(x3 + (256L*x2) + (32768L*x1) + (2097152L*x0)), to_float_mask(tmp41));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp42(), to_float_mask(tmp41));
                            auto tmp45 = at::vec::maximum(tmp44, tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = masked_load(in_out_ptr0 + static_cast<long>(128L + x3 + (256L*x2) + (32768L*x1) + (2097152L*x0)), to_float_mask(tmp46));
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
                                auto tmp57 = masked_load(in_out_ptr0 + static_cast<long>(16256L + x3 + (256L*x2) + (32768L*x1) + (2097152L*x0)), to_float_mask(tmp55));
                                return tmp57;
                            }
                            ;
                            auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp56(), to_float_mask(tmp55));
                            auto tmp59 = at::vec::maximum(tmp58, tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = masked_load(in_out_ptr0 + static_cast<long>(16384L + x3 + (256L*x2) + (32768L*x1) + (2097152L*x0)), to_float_mask(tmp60));
                                return tmp62;
                            }
                            ;
                            auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp61(), to_float_mask(tmp60));
                            auto tmp64 = at::vec::maximum(tmp63, tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = masked_load(in_out_ptr0 + static_cast<long>(16512L + x3 + (256L*x2) + (32768L*x1) + (2097152L*x0)), to_float_mask(tmp65));
                                return tmp67;
                            }
                            ;
                            auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp66(), to_float_mask(tmp65));
                            auto tmp69 = at::vec::maximum(tmp68, tmp64);
                            tmp69.store(out_ptr0 + static_cast<long>(x3 + (128L*x2) + (8192L*x1) + (524288L*x0)));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x2) + (524288L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(64L + x1 + (128L*x2) + (524288L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (524288L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (128L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(64L + x2 + (128L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x2 + (128L*x1) + (524288L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x2) + (524288L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(64L + x1 + (128L*x2) + (524288L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (524288L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (128L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(64L + x2 + (128L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x2 + (128L*x1) + (524288L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_13 = async_compile.cpp('''
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
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_15 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x2) + (524288L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(64L + x1 + (128L*x2) + (524288L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (524288L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (128L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(64L + x2 + (128L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x2 + (128L*x1) + (524288L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_18 = async_compile.cpp('''
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
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_19 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(4096L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x2) + (1048576L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(128L + x1 + (256L*x2) + (1048576L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_avg_pool2d_mul_sum_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (1048576L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (256L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(128L + x2 + (256L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (256L*x1) + (1048576L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (524288L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-1L) + (2L*x1));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(64);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<int>((-1L) + (2L*x2));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(out_ptr1 + static_cast<long>((-8320L) + x3 + (256L*x2) + (16384L*x1) + (524288L*x0)), to_float_mask(tmp10));
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
                                auto tmp20 = masked_load(out_ptr1 + static_cast<long>((-8192L) + x3 + (256L*x2) + (16384L*x1) + (524288L*x0)), to_float_mask(tmp18));
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
                                auto tmp29 = masked_load(out_ptr1 + static_cast<long>((-8064L) + x3 + (256L*x2) + (16384L*x1) + (524288L*x0)), to_float_mask(tmp27));
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
                                auto tmp38 = masked_load(out_ptr1 + static_cast<long>((-128L) + x3 + (256L*x2) + (16384L*x1) + (524288L*x0)), to_float_mask(tmp36));
                                return tmp38;
                            }
                            ;
                            auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp37(), to_float_mask(tmp36));
                            auto tmp40 = tmp39 + tmp31;
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(out_ptr1 + static_cast<long>(x3 + (256L*x2) + (16384L*x1) + (524288L*x0)), to_float_mask(tmp41));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp41));
                            auto tmp45 = tmp44 + tmp40;
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = masked_load(out_ptr1 + static_cast<long>(128L + x3 + (256L*x2) + (16384L*x1) + (524288L*x0)), to_float_mask(tmp46));
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
                                auto tmp57 = masked_load(out_ptr1 + static_cast<long>(8064L + x3 + (256L*x2) + (16384L*x1) + (524288L*x0)), to_float_mask(tmp55));
                                return tmp57;
                            }
                            ;
                            auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp56(), to_float_mask(tmp55));
                            auto tmp59 = tmp58 + tmp50;
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = masked_load(out_ptr1 + static_cast<long>(8192L + x3 + (256L*x2) + (16384L*x1) + (524288L*x0)), to_float_mask(tmp60));
                                return tmp62;
                            }
                            ;
                            auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp61(), to_float_mask(tmp60));
                            auto tmp64 = tmp63 + tmp59;
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = masked_load(out_ptr1 + static_cast<long>(8320L + x3 + (256L*x2) + (16384L*x1) + (524288L*x0)), to_float_mask(tmp65));
                                return tmp67;
                            }
                            ;
                            auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp66(), to_float_mask(tmp65));
                            auto tmp69 = tmp68 + tmp64;
                            auto tmp70 = static_cast<int>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<int>(65);
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
                                auto tmp83 = static_cast<int>(64);
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
                                auto tmp104 = static_cast<int>(64);
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
                                auto tmp126 = static_cast<int>(64);
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
                                auto tmp148 = static_cast<int>(64);
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
                                auto tmp167 = static_cast<int>(64);
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
                                auto tmp186 = static_cast<int>(64);
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
                                auto tmp208 = static_cast<int>(64);
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
                                auto tmp227 = static_cast<int>(64);
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
                                auto tmp246 = static_cast<int>(64);
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
                            tmp261.store(out_ptr2 + static_cast<long>(x3 + (128L*x2) + (4096L*x1) + (131072L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_23 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (32768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x2 + (512L*x1) + (32768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16384L + x2 + (512L*x1) + (32768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16640L + x2 + (512L*x1) + (32768L*x0)));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_24 = async_compile.cpp('''
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
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_25 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_26 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x2) + (262144L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(128L + x1 + (256L*x2) + (262144L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (262144L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (256L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(128L + x2 + (256L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (256L*x1) + (262144L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_29 = async_compile.cpp('''
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
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_30 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_31 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x2) + (262144L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(128L + x1 + (256L*x2) + (262144L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (262144L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (256L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(128L + x2 + (256L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (256L*x1) + (262144L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_34 = async_compile.cpp('''
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
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_36 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x2) + (262144L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(128L + x1 + (256L*x2) + (262144L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (262144L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (256L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(128L + x2 + (256L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (256L*x1) + (262144L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_39 = async_compile.cpp('''
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
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_40 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (524288L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_avg_pool2d_mul_sum_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (524288L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (524288L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (262144L*x0)));
                    }
                }
            }
        }
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
                            auto tmp0 = c10::convert<int>((-1L) + (2L*x1));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(32);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<int>((-1L) + (2L*x2));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(out_ptr1 + static_cast<long>((-8448L) + x3 + (512L*x2) + (16384L*x1) + (262144L*x0)), to_float_mask(tmp10));
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
                                auto tmp20 = masked_load(out_ptr1 + static_cast<long>((-8192L) + x3 + (512L*x2) + (16384L*x1) + (262144L*x0)), to_float_mask(tmp18));
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
                                auto tmp29 = masked_load(out_ptr1 + static_cast<long>((-7936L) + x3 + (512L*x2) + (16384L*x1) + (262144L*x0)), to_float_mask(tmp27));
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
                                auto tmp38 = masked_load(out_ptr1 + static_cast<long>((-256L) + x3 + (512L*x2) + (16384L*x1) + (262144L*x0)), to_float_mask(tmp36));
                                return tmp38;
                            }
                            ;
                            auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp37(), to_float_mask(tmp36));
                            auto tmp40 = tmp39 + tmp31;
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(out_ptr1 + static_cast<long>(x3 + (512L*x2) + (16384L*x1) + (262144L*x0)), to_float_mask(tmp41));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp41));
                            auto tmp45 = tmp44 + tmp40;
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = masked_load(out_ptr1 + static_cast<long>(256L + x3 + (512L*x2) + (16384L*x1) + (262144L*x0)), to_float_mask(tmp46));
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
                                auto tmp57 = masked_load(out_ptr1 + static_cast<long>(7936L + x3 + (512L*x2) + (16384L*x1) + (262144L*x0)), to_float_mask(tmp55));
                                return tmp57;
                            }
                            ;
                            auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp56(), to_float_mask(tmp55));
                            auto tmp59 = tmp58 + tmp50;
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = masked_load(out_ptr1 + static_cast<long>(8192L + x3 + (512L*x2) + (16384L*x1) + (262144L*x0)), to_float_mask(tmp60));
                                return tmp62;
                            }
                            ;
                            auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp61(), to_float_mask(tmp60));
                            auto tmp64 = tmp63 + tmp59;
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = masked_load(out_ptr1 + static_cast<long>(8448L + x3 + (512L*x2) + (16384L*x1) + (262144L*x0)), to_float_mask(tmp65));
                                return tmp67;
                            }
                            ;
                            auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp66(), to_float_mask(tmp65));
                            auto tmp69 = tmp68 + tmp64;
                            auto tmp70 = static_cast<int>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<int>(33);
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
                                auto tmp83 = static_cast<int>(32);
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
                                auto tmp104 = static_cast<int>(32);
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
                                auto tmp126 = static_cast<int>(32);
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
                                auto tmp148 = static_cast<int>(32);
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
                                auto tmp167 = static_cast<int>(32);
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
                                auto tmp186 = static_cast<int>(32);
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
                                auto tmp208 = static_cast<int>(32);
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
                                auto tmp227 = static_cast<int>(32);
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
                                auto tmp246 = static_cast<int>(32);
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
                            tmp261.store(out_ptr2 + static_cast<long>(x3 + (256L*x2) + (4096L*x1) + (65536L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*x1) + (32768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x2 + (1024L*x1) + (32768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16384L + x2 + (1024L*x1) + (32768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16896L + x2 + (1024L*x1) + (32768L*x0)));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_45 = async_compile.cpp('''
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
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_46 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_47 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_50 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_51 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_52 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_55 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_57 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_60 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_61 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_62 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_65 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_66 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_67 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_70 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_72 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_75 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_76 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_77 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_80 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_81 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_82 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_85 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_87 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_90 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_91 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_92 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_95 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_96 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_97 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_100 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_102 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_105 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_106 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_107 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_110 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_111 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_112 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_115 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_116 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_117 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_120 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_121 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_122 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_124 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_125 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_126 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_127 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_129 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_130 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_131 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_132 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_133 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_134 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_135 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_136 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_137 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_138 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_139 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_140 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_141 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_142 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_143 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_145 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_146 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_147 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_148 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_149 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_150 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_151 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_152 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (131072L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_153 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_154 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (131072L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_155 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_156 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_157 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x2) + (262144L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(512L + x1 + (1024L*x2) + (262144L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_158 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_avg_pool2d_mul_sum_159 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1024L*x1) + (262144L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (1024L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(512L + x2 + (1024L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x2 + (1024L*x1) + (262144L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                    }
                }
            }
        }
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
                            auto tmp0 = c10::convert<int>((-1L) + (2L*x1));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(16);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<int>((-1L) + (2L*x2));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(out_ptr1 + static_cast<long>((-8704L) + x3 + (1024L*x2) + (16384L*x1) + (131072L*x0)), to_float_mask(tmp10));
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
                                auto tmp20 = masked_load(out_ptr1 + static_cast<long>((-8192L) + x3 + (1024L*x2) + (16384L*x1) + (131072L*x0)), to_float_mask(tmp18));
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
                                auto tmp29 = masked_load(out_ptr1 + static_cast<long>((-7680L) + x3 + (1024L*x2) + (16384L*x1) + (131072L*x0)), to_float_mask(tmp27));
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
                                auto tmp38 = masked_load(out_ptr1 + static_cast<long>((-512L) + x3 + (1024L*x2) + (16384L*x1) + (131072L*x0)), to_float_mask(tmp36));
                                return tmp38;
                            }
                            ;
                            auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp37(), to_float_mask(tmp36));
                            auto tmp40 = tmp39 + tmp31;
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(out_ptr1 + static_cast<long>(x3 + (1024L*x2) + (16384L*x1) + (131072L*x0)), to_float_mask(tmp41));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp41));
                            auto tmp45 = tmp44 + tmp40;
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = masked_load(out_ptr1 + static_cast<long>(512L + x3 + (1024L*x2) + (16384L*x1) + (131072L*x0)), to_float_mask(tmp46));
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
                                auto tmp57 = masked_load(out_ptr1 + static_cast<long>(7680L + x3 + (1024L*x2) + (16384L*x1) + (131072L*x0)), to_float_mask(tmp55));
                                return tmp57;
                            }
                            ;
                            auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp56(), to_float_mask(tmp55));
                            auto tmp59 = tmp58 + tmp50;
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = masked_load(out_ptr1 + static_cast<long>(8192L + x3 + (1024L*x2) + (16384L*x1) + (131072L*x0)), to_float_mask(tmp60));
                                return tmp62;
                            }
                            ;
                            auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp61(), to_float_mask(tmp60));
                            auto tmp64 = tmp63 + tmp59;
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = masked_load(out_ptr1 + static_cast<long>(8704L + x3 + (1024L*x2) + (16384L*x1) + (131072L*x0)), to_float_mask(tmp65));
                                return tmp67;
                            }
                            ;
                            auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp66(), to_float_mask(tmp65));
                            auto tmp69 = tmp68 + tmp64;
                            auto tmp70 = static_cast<int>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<int>(17);
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
                                auto tmp83 = static_cast<int>(16);
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
                                auto tmp104 = static_cast<int>(16);
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
                                auto tmp126 = static_cast<int>(16);
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
                                auto tmp148 = static_cast<int>(16);
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
                                auto tmp167 = static_cast<int>(16);
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
                                auto tmp186 = static_cast<int>(16);
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
                                auto tmp208 = static_cast<int>(16);
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
                                auto tmp227 = static_cast<int>(16);
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
                                auto tmp246 = static_cast<int>(16);
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
                            tmp261.store(out_ptr2 + static_cast<long>(x3 + (512L*x2) + (4096L*x1) + (32768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_160 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (2048L*x1) + (32768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + x2 + (2048L*x1) + (32768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16384L + x2 + (2048L*x1) + (32768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(17408L + x2 + (2048L*x1) + (32768L*x0)));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_161 = async_compile.cpp('''
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
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_162 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_163 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x2) + (65536L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(512L + x1 + (1024L*x2) + (65536L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_164 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_165 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1024L*x1) + (65536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (1024L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(512L + x2 + (1024L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x2 + (1024L*x1) + (65536L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_166 = async_compile.cpp('''
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
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_167 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_168 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x2) + (65536L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(512L + x1 + (1024L*x2) + (65536L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_169 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_mul_sum_170 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1024L*x1) + (65536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (1024L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(512L + x2 + (1024L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x2 + (1024L*x1) + (65536L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp7 = tmp2 / tmp3;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_171 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (2048L*x2) + (131072L*x0)));
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
                    auto tmp1 = static_cast<float>(64.0);
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (64, 128, 1, 1), (128, 1, 1, 1))
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
    assert_size_stride(arg24_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg25_1, (256, ), (1, ))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg28_1, (64, ), (1, ))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (128, ), (1, ))
    assert_size_stride(arg33_1, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg34_1, (32, ), (1, ))
    assert_size_stride(arg35_1, (32, ), (1, ))
    assert_size_stride(arg36_1, (32, ), (1, ))
    assert_size_stride(arg37_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (256, ), (1, ))
    assert_size_stride(arg42_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (64, ), (1, ))
    assert_size_stride(arg45_1, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg49_1, (32, ), (1, ))
    assert_size_stride(arg50_1, (32, ), (1, ))
    assert_size_stride(arg51_1, (32, ), (1, ))
    assert_size_stride(arg52_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg53_1, (128, ), (1, ))
    assert_size_stride(arg54_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg55_1, (256, ), (1, ))
    assert_size_stride(arg56_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (128, ), (1, ))
    assert_size_stride(arg60_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg61_1, (256, ), (1, ))
    assert_size_stride(arg62_1, (256, ), (1, ))
    assert_size_stride(arg63_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg64_1, (64, ), (1, ))
    assert_size_stride(arg65_1, (64, ), (1, ))
    assert_size_stride(arg66_1, (64, ), (1, ))
    assert_size_stride(arg67_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg68_1, (256, ), (1, ))
    assert_size_stride(arg69_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, ), (1, ))
    assert_size_stride(arg72_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg76_1, (128, ), (1, ))
    assert_size_stride(arg77_1, (128, ), (1, ))
    assert_size_stride(arg78_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg79_1, (256, ), (1, ))
    assert_size_stride(arg80_1, (256, ), (1, ))
    assert_size_stride(arg81_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg82_1, (64, ), (1, ))
    assert_size_stride(arg83_1, (64, ), (1, ))
    assert_size_stride(arg84_1, (64, ), (1, ))
    assert_size_stride(arg85_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg88_1, (512, ), (1, ))
    assert_size_stride(arg89_1, (512, ), (1, ))
    assert_size_stride(arg90_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg91_1, (128, ), (1, ))
    assert_size_stride(arg92_1, (128, ), (1, ))
    assert_size_stride(arg93_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg95_1, (256, ), (1, ))
    assert_size_stride(arg96_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg97_1, (64, ), (1, ))
    assert_size_stride(arg98_1, (64, ), (1, ))
    assert_size_stride(arg99_1, (64, ), (1, ))
    assert_size_stride(arg100_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg101_1, (256, ), (1, ))
    assert_size_stride(arg102_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg103_1, (512, ), (1, ))
    assert_size_stride(arg104_1, (512, ), (1, ))
    assert_size_stride(arg105_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg106_1, (128, ), (1, ))
    assert_size_stride(arg107_1, (128, ), (1, ))
    assert_size_stride(arg108_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg109_1, (256, ), (1, ))
    assert_size_stride(arg110_1, (256, ), (1, ))
    assert_size_stride(arg111_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg112_1, (64, ), (1, ))
    assert_size_stride(arg113_1, (64, ), (1, ))
    assert_size_stride(arg114_1, (64, ), (1, ))
    assert_size_stride(arg115_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg116_1, (256, ), (1, ))
    assert_size_stride(arg117_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg118_1, (512, ), (1, ))
    assert_size_stride(arg119_1, (512, ), (1, ))
    assert_size_stride(arg120_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg121_1, (256, ), (1, ))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg124_1, (512, ), (1, ))
    assert_size_stride(arg125_1, (512, ), (1, ))
    assert_size_stride(arg126_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg127_1, (128, ), (1, ))
    assert_size_stride(arg128_1, (128, ), (1, ))
    assert_size_stride(arg129_1, (128, ), (1, ))
    assert_size_stride(arg130_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg131_1, (512, ), (1, ))
    assert_size_stride(arg132_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg136_1, (1024, ), (1, ))
    assert_size_stride(arg137_1, (1024, ), (1, ))
    assert_size_stride(arg138_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg139_1, (256, ), (1, ))
    assert_size_stride(arg140_1, (256, ), (1, ))
    assert_size_stride(arg141_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (512, ), (1, ))
    assert_size_stride(arg144_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg145_1, (128, ), (1, ))
    assert_size_stride(arg146_1, (128, ), (1, ))
    assert_size_stride(arg147_1, (128, ), (1, ))
    assert_size_stride(arg148_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg149_1, (512, ), (1, ))
    assert_size_stride(arg150_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg151_1, (1024, ), (1, ))
    assert_size_stride(arg152_1, (1024, ), (1, ))
    assert_size_stride(arg153_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg154_1, (256, ), (1, ))
    assert_size_stride(arg155_1, (256, ), (1, ))
    assert_size_stride(arg156_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg157_1, (512, ), (1, ))
    assert_size_stride(arg158_1, (512, ), (1, ))
    assert_size_stride(arg159_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg160_1, (128, ), (1, ))
    assert_size_stride(arg161_1, (128, ), (1, ))
    assert_size_stride(arg162_1, (128, ), (1, ))
    assert_size_stride(arg163_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg164_1, (512, ), (1, ))
    assert_size_stride(arg165_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg166_1, (1024, ), (1, ))
    assert_size_stride(arg167_1, (1024, ), (1, ))
    assert_size_stride(arg168_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg169_1, (256, ), (1, ))
    assert_size_stride(arg170_1, (256, ), (1, ))
    assert_size_stride(arg171_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg172_1, (512, ), (1, ))
    assert_size_stride(arg173_1, (512, ), (1, ))
    assert_size_stride(arg174_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg175_1, (128, ), (1, ))
    assert_size_stride(arg176_1, (128, ), (1, ))
    assert_size_stride(arg177_1, (128, ), (1, ))
    assert_size_stride(arg178_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg179_1, (512, ), (1, ))
    assert_size_stride(arg180_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg181_1, (1024, ), (1, ))
    assert_size_stride(arg182_1, (1024, ), (1, ))
    assert_size_stride(arg183_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg184_1, (256, ), (1, ))
    assert_size_stride(arg185_1, (256, ), (1, ))
    assert_size_stride(arg186_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg187_1, (512, ), (1, ))
    assert_size_stride(arg188_1, (512, ), (1, ))
    assert_size_stride(arg189_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg190_1, (128, ), (1, ))
    assert_size_stride(arg191_1, (128, ), (1, ))
    assert_size_stride(arg192_1, (128, ), (1, ))
    assert_size_stride(arg193_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg196_1, (1024, ), (1, ))
    assert_size_stride(arg197_1, (1024, ), (1, ))
    assert_size_stride(arg198_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg199_1, (256, ), (1, ))
    assert_size_stride(arg200_1, (256, ), (1, ))
    assert_size_stride(arg201_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg202_1, (512, ), (1, ))
    assert_size_stride(arg203_1, (512, ), (1, ))
    assert_size_stride(arg204_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg205_1, (128, ), (1, ))
    assert_size_stride(arg206_1, (128, ), (1, ))
    assert_size_stride(arg207_1, (128, ), (1, ))
    assert_size_stride(arg208_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg209_1, (512, ), (1, ))
    assert_size_stride(arg210_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg211_1, (1024, ), (1, ))
    assert_size_stride(arg212_1, (1024, ), (1, ))
    assert_size_stride(arg213_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg214_1, (256, ), (1, ))
    assert_size_stride(arg215_1, (256, ), (1, ))
    assert_size_stride(arg216_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg217_1, (512, ), (1, ))
    assert_size_stride(arg218_1, (512, ), (1, ))
    assert_size_stride(arg219_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg220_1, (128, ), (1, ))
    assert_size_stride(arg221_1, (128, ), (1, ))
    assert_size_stride(arg222_1, (128, ), (1, ))
    assert_size_stride(arg223_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg224_1, (512, ), (1, ))
    assert_size_stride(arg225_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg226_1, (1024, ), (1, ))
    assert_size_stride(arg227_1, (1024, ), (1, ))
    assert_size_stride(arg228_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg229_1, (256, ), (1, ))
    assert_size_stride(arg230_1, (256, ), (1, ))
    assert_size_stride(arg231_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg232_1, (512, ), (1, ))
    assert_size_stride(arg233_1, (512, ), (1, ))
    assert_size_stride(arg234_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg235_1, (128, ), (1, ))
    assert_size_stride(arg236_1, (128, ), (1, ))
    assert_size_stride(arg237_1, (128, ), (1, ))
    assert_size_stride(arg238_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg239_1, (512, ), (1, ))
    assert_size_stride(arg240_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg241_1, (1024, ), (1, ))
    assert_size_stride(arg242_1, (1024, ), (1, ))
    assert_size_stride(arg243_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg244_1, (256, ), (1, ))
    assert_size_stride(arg245_1, (256, ), (1, ))
    assert_size_stride(arg246_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg247_1, (512, ), (1, ))
    assert_size_stride(arg248_1, (512, ), (1, ))
    assert_size_stride(arg249_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg250_1, (128, ), (1, ))
    assert_size_stride(arg251_1, (128, ), (1, ))
    assert_size_stride(arg252_1, (128, ), (1, ))
    assert_size_stride(arg253_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg254_1, (512, ), (1, ))
    assert_size_stride(arg255_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg256_1, (1024, ), (1, ))
    assert_size_stride(arg257_1, (1024, ), (1, ))
    assert_size_stride(arg258_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg259_1, (256, ), (1, ))
    assert_size_stride(arg260_1, (256, ), (1, ))
    assert_size_stride(arg261_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg262_1, (512, ), (1, ))
    assert_size_stride(arg263_1, (512, ), (1, ))
    assert_size_stride(arg264_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg265_1, (128, ), (1, ))
    assert_size_stride(arg266_1, (128, ), (1, ))
    assert_size_stride(arg267_1, (128, ), (1, ))
    assert_size_stride(arg268_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg269_1, (512, ), (1, ))
    assert_size_stride(arg270_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg271_1, (1024, ), (1, ))
    assert_size_stride(arg272_1, (1024, ), (1, ))
    assert_size_stride(arg273_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg274_1, (256, ), (1, ))
    assert_size_stride(arg275_1, (256, ), (1, ))
    assert_size_stride(arg276_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg277_1, (512, ), (1, ))
    assert_size_stride(arg278_1, (512, ), (1, ))
    assert_size_stride(arg279_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg280_1, (128, ), (1, ))
    assert_size_stride(arg281_1, (128, ), (1, ))
    assert_size_stride(arg282_1, (128, ), (1, ))
    assert_size_stride(arg283_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg284_1, (512, ), (1, ))
    assert_size_stride(arg285_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg286_1, (1024, ), (1, ))
    assert_size_stride(arg287_1, (1024, ), (1, ))
    assert_size_stride(arg288_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg289_1, (256, ), (1, ))
    assert_size_stride(arg290_1, (256, ), (1, ))
    assert_size_stride(arg291_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg292_1, (512, ), (1, ))
    assert_size_stride(arg293_1, (512, ), (1, ))
    assert_size_stride(arg294_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg295_1, (128, ), (1, ))
    assert_size_stride(arg296_1, (128, ), (1, ))
    assert_size_stride(arg297_1, (128, ), (1, ))
    assert_size_stride(arg298_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg299_1, (512, ), (1, ))
    assert_size_stride(arg300_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg301_1, (1024, ), (1, ))
    assert_size_stride(arg302_1, (1024, ), (1, ))
    assert_size_stride(arg303_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg304_1, (256, ), (1, ))
    assert_size_stride(arg305_1, (256, ), (1, ))
    assert_size_stride(arg306_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg307_1, (512, ), (1, ))
    assert_size_stride(arg308_1, (512, ), (1, ))
    assert_size_stride(arg309_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg310_1, (128, ), (1, ))
    assert_size_stride(arg311_1, (128, ), (1, ))
    assert_size_stride(arg312_1, (128, ), (1, ))
    assert_size_stride(arg313_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg314_1, (512, ), (1, ))
    assert_size_stride(arg315_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg316_1, (1024, ), (1, ))
    assert_size_stride(arg317_1, (1024, ), (1, ))
    assert_size_stride(arg318_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg319_1, (256, ), (1, ))
    assert_size_stride(arg320_1, (256, ), (1, ))
    assert_size_stride(arg321_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg322_1, (512, ), (1, ))
    assert_size_stride(arg323_1, (512, ), (1, ))
    assert_size_stride(arg324_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg325_1, (128, ), (1, ))
    assert_size_stride(arg326_1, (128, ), (1, ))
    assert_size_stride(arg327_1, (128, ), (1, ))
    assert_size_stride(arg328_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg329_1, (512, ), (1, ))
    assert_size_stride(arg330_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg331_1, (1024, ), (1, ))
    assert_size_stride(arg332_1, (1024, ), (1, ))
    assert_size_stride(arg333_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg334_1, (256, ), (1, ))
    assert_size_stride(arg335_1, (256, ), (1, ))
    assert_size_stride(arg336_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg337_1, (512, ), (1, ))
    assert_size_stride(arg338_1, (512, ), (1, ))
    assert_size_stride(arg339_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg340_1, (128, ), (1, ))
    assert_size_stride(arg341_1, (128, ), (1, ))
    assert_size_stride(arg342_1, (128, ), (1, ))
    assert_size_stride(arg343_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg344_1, (512, ), (1, ))
    assert_size_stride(arg345_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg346_1, (1024, ), (1, ))
    assert_size_stride(arg347_1, (1024, ), (1, ))
    assert_size_stride(arg348_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg349_1, (256, ), (1, ))
    assert_size_stride(arg350_1, (256, ), (1, ))
    assert_size_stride(arg351_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg352_1, (512, ), (1, ))
    assert_size_stride(arg353_1, (512, ), (1, ))
    assert_size_stride(arg354_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg355_1, (128, ), (1, ))
    assert_size_stride(arg356_1, (128, ), (1, ))
    assert_size_stride(arg357_1, (128, ), (1, ))
    assert_size_stride(arg358_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg359_1, (512, ), (1, ))
    assert_size_stride(arg360_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg361_1, (1024, ), (1, ))
    assert_size_stride(arg362_1, (1024, ), (1, ))
    assert_size_stride(arg363_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg364_1, (256, ), (1, ))
    assert_size_stride(arg365_1, (256, ), (1, ))
    assert_size_stride(arg366_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg367_1, (512, ), (1, ))
    assert_size_stride(arg368_1, (512, ), (1, ))
    assert_size_stride(arg369_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg370_1, (128, ), (1, ))
    assert_size_stride(arg371_1, (128, ), (1, ))
    assert_size_stride(arg372_1, (128, ), (1, ))
    assert_size_stride(arg373_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg374_1, (512, ), (1, ))
    assert_size_stride(arg375_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg376_1, (1024, ), (1, ))
    assert_size_stride(arg377_1, (1024, ), (1, ))
    assert_size_stride(arg378_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg379_1, (256, ), (1, ))
    assert_size_stride(arg380_1, (256, ), (1, ))
    assert_size_stride(arg381_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg382_1, (512, ), (1, ))
    assert_size_stride(arg383_1, (512, ), (1, ))
    assert_size_stride(arg384_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg385_1, (128, ), (1, ))
    assert_size_stride(arg386_1, (128, ), (1, ))
    assert_size_stride(arg387_1, (128, ), (1, ))
    assert_size_stride(arg388_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg389_1, (512, ), (1, ))
    assert_size_stride(arg390_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg391_1, (1024, ), (1, ))
    assert_size_stride(arg392_1, (1024, ), (1, ))
    assert_size_stride(arg393_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg394_1, (256, ), (1, ))
    assert_size_stride(arg395_1, (256, ), (1, ))
    assert_size_stride(arg396_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg397_1, (512, ), (1, ))
    assert_size_stride(arg398_1, (512, ), (1, ))
    assert_size_stride(arg399_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg400_1, (128, ), (1, ))
    assert_size_stride(arg401_1, (128, ), (1, ))
    assert_size_stride(arg402_1, (128, ), (1, ))
    assert_size_stride(arg403_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg404_1, (512, ), (1, ))
    assert_size_stride(arg405_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg406_1, (1024, ), (1, ))
    assert_size_stride(arg407_1, (1024, ), (1, ))
    assert_size_stride(arg408_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg409_1, (256, ), (1, ))
    assert_size_stride(arg410_1, (256, ), (1, ))
    assert_size_stride(arg411_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg412_1, (512, ), (1, ))
    assert_size_stride(arg413_1, (512, ), (1, ))
    assert_size_stride(arg414_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg415_1, (128, ), (1, ))
    assert_size_stride(arg416_1, (128, ), (1, ))
    assert_size_stride(arg417_1, (128, ), (1, ))
    assert_size_stride(arg418_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg419_1, (512, ), (1, ))
    assert_size_stride(arg420_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg421_1, (1024, ), (1, ))
    assert_size_stride(arg422_1, (1024, ), (1, ))
    assert_size_stride(arg423_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg424_1, (256, ), (1, ))
    assert_size_stride(arg425_1, (256, ), (1, ))
    assert_size_stride(arg426_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg427_1, (512, ), (1, ))
    assert_size_stride(arg428_1, (512, ), (1, ))
    assert_size_stride(arg429_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg430_1, (128, ), (1, ))
    assert_size_stride(arg431_1, (128, ), (1, ))
    assert_size_stride(arg432_1, (128, ), (1, ))
    assert_size_stride(arg433_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg434_1, (512, ), (1, ))
    assert_size_stride(arg435_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg436_1, (1024, ), (1, ))
    assert_size_stride(arg437_1, (1024, ), (1, ))
    assert_size_stride(arg438_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg439_1, (256, ), (1, ))
    assert_size_stride(arg440_1, (256, ), (1, ))
    assert_size_stride(arg441_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg442_1, (512, ), (1, ))
    assert_size_stride(arg443_1, (512, ), (1, ))
    assert_size_stride(arg444_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg445_1, (128, ), (1, ))
    assert_size_stride(arg446_1, (128, ), (1, ))
    assert_size_stride(arg447_1, (128, ), (1, ))
    assert_size_stride(arg448_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg449_1, (512, ), (1, ))
    assert_size_stride(arg450_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg451_1, (1024, ), (1, ))
    assert_size_stride(arg452_1, (1024, ), (1, ))
    assert_size_stride(arg453_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg454_1, (256, ), (1, ))
    assert_size_stride(arg455_1, (256, ), (1, ))
    assert_size_stride(arg456_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg457_1, (512, ), (1, ))
    assert_size_stride(arg458_1, (512, ), (1, ))
    assert_size_stride(arg459_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg460_1, (128, ), (1, ))
    assert_size_stride(arg461_1, (128, ), (1, ))
    assert_size_stride(arg462_1, (128, ), (1, ))
    assert_size_stride(arg463_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg464_1, (512, ), (1, ))
    assert_size_stride(arg465_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg466_1, (1024, ), (1, ))
    assert_size_stride(arg467_1, (1024, ), (1, ))
    assert_size_stride(arg468_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg469_1, (512, ), (1, ))
    assert_size_stride(arg470_1, (512, ), (1, ))
    assert_size_stride(arg471_1, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg472_1, (1024, ), (1, ))
    assert_size_stride(arg473_1, (1024, ), (1, ))
    assert_size_stride(arg474_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg475_1, (256, ), (1, ))
    assert_size_stride(arg476_1, (256, ), (1, ))
    assert_size_stride(arg477_1, (256, ), (1, ))
    assert_size_stride(arg478_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg479_1, (1024, ), (1, ))
    assert_size_stride(arg480_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg481_1, (2048, ), (1, ))
    assert_size_stride(arg482_1, (2048, ), (1, ))
    assert_size_stride(arg483_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg484_1, (2048, ), (1, ))
    assert_size_stride(arg485_1, (2048, ), (1, ))
    assert_size_stride(arg486_1, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg487_1, (512, ), (1, ))
    assert_size_stride(arg488_1, (512, ), (1, ))
    assert_size_stride(arg489_1, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg490_1, (1024, ), (1, ))
    assert_size_stride(arg491_1, (1024, ), (1, ))
    assert_size_stride(arg492_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg493_1, (256, ), (1, ))
    assert_size_stride(arg494_1, (256, ), (1, ))
    assert_size_stride(arg495_1, (256, ), (1, ))
    assert_size_stride(arg496_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg497_1, (1024, ), (1, ))
    assert_size_stride(arg498_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg499_1, (2048, ), (1, ))
    assert_size_stride(arg500_1, (2048, ), (1, ))
    assert_size_stride(arg501_1, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg502_1, (512, ), (1, ))
    assert_size_stride(arg503_1, (512, ), (1, ))
    assert_size_stride(arg504_1, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg505_1, (1024, ), (1, ))
    assert_size_stride(arg506_1, (1024, ), (1, ))
    assert_size_stride(arg507_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg508_1, (256, ), (1, ))
    assert_size_stride(arg509_1, (256, ), (1, ))
    assert_size_stride(arg510_1, (256, ), (1, ))
    assert_size_stride(arg511_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg512_1, (1024, ), (1, ))
    assert_size_stride(arg513_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg514_1, (2048, ), (1, ))
    assert_size_stride(arg515_1, (2048, ), (1, ))
    assert_size_stride(arg516_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg517_1, (1000, ), (1, ))
    assert_size_stride(arg518_1, (64, ), (1, ))
    assert_size_stride(arg519_1, (64, ), (1, ))
    assert_size_stride(arg520_1, (), ())
    assert_size_stride(arg521_1, (64, ), (1, ))
    assert_size_stride(arg522_1, (64, ), (1, ))
    assert_size_stride(arg523_1, (), ())
    assert_size_stride(arg524_1, (128, ), (1, ))
    assert_size_stride(arg525_1, (128, ), (1, ))
    assert_size_stride(arg526_1, (), ())
    assert_size_stride(arg527_1, (64, ), (1, ))
    assert_size_stride(arg528_1, (64, ), (1, ))
    assert_size_stride(arg529_1, (), ())
    assert_size_stride(arg530_1, (128, ), (1, ))
    assert_size_stride(arg531_1, (128, ), (1, ))
    assert_size_stride(arg532_1, (), ())
    assert_size_stride(arg533_1, (32, ), (1, ))
    assert_size_stride(arg534_1, (32, ), (1, ))
    assert_size_stride(arg535_1, (), ())
    assert_size_stride(arg536_1, (256, ), (1, ))
    assert_size_stride(arg537_1, (256, ), (1, ))
    assert_size_stride(arg538_1, (), ())
    assert_size_stride(arg539_1, (256, ), (1, ))
    assert_size_stride(arg540_1, (256, ), (1, ))
    assert_size_stride(arg541_1, (), ())
    assert_size_stride(arg542_1, (64, ), (1, ))
    assert_size_stride(arg543_1, (64, ), (1, ))
    assert_size_stride(arg544_1, (), ())
    assert_size_stride(arg545_1, (128, ), (1, ))
    assert_size_stride(arg546_1, (128, ), (1, ))
    assert_size_stride(arg547_1, (), ())
    assert_size_stride(arg548_1, (32, ), (1, ))
    assert_size_stride(arg549_1, (32, ), (1, ))
    assert_size_stride(arg550_1, (), ())
    assert_size_stride(arg551_1, (256, ), (1, ))
    assert_size_stride(arg552_1, (256, ), (1, ))
    assert_size_stride(arg553_1, (), ())
    assert_size_stride(arg554_1, (64, ), (1, ))
    assert_size_stride(arg555_1, (64, ), (1, ))
    assert_size_stride(arg556_1, (), ())
    assert_size_stride(arg557_1, (128, ), (1, ))
    assert_size_stride(arg558_1, (128, ), (1, ))
    assert_size_stride(arg559_1, (), ())
    assert_size_stride(arg560_1, (32, ), (1, ))
    assert_size_stride(arg561_1, (32, ), (1, ))
    assert_size_stride(arg562_1, (), ())
    assert_size_stride(arg563_1, (256, ), (1, ))
    assert_size_stride(arg564_1, (256, ), (1, ))
    assert_size_stride(arg565_1, (), ())
    assert_size_stride(arg566_1, (128, ), (1, ))
    assert_size_stride(arg567_1, (128, ), (1, ))
    assert_size_stride(arg568_1, (), ())
    assert_size_stride(arg569_1, (256, ), (1, ))
    assert_size_stride(arg570_1, (256, ), (1, ))
    assert_size_stride(arg571_1, (), ())
    assert_size_stride(arg572_1, (64, ), (1, ))
    assert_size_stride(arg573_1, (64, ), (1, ))
    assert_size_stride(arg574_1, (), ())
    assert_size_stride(arg575_1, (512, ), (1, ))
    assert_size_stride(arg576_1, (512, ), (1, ))
    assert_size_stride(arg577_1, (), ())
    assert_size_stride(arg578_1, (512, ), (1, ))
    assert_size_stride(arg579_1, (512, ), (1, ))
    assert_size_stride(arg580_1, (), ())
    assert_size_stride(arg581_1, (128, ), (1, ))
    assert_size_stride(arg582_1, (128, ), (1, ))
    assert_size_stride(arg583_1, (), ())
    assert_size_stride(arg584_1, (256, ), (1, ))
    assert_size_stride(arg585_1, (256, ), (1, ))
    assert_size_stride(arg586_1, (), ())
    assert_size_stride(arg587_1, (64, ), (1, ))
    assert_size_stride(arg588_1, (64, ), (1, ))
    assert_size_stride(arg589_1, (), ())
    assert_size_stride(arg590_1, (512, ), (1, ))
    assert_size_stride(arg591_1, (512, ), (1, ))
    assert_size_stride(arg592_1, (), ())
    assert_size_stride(arg593_1, (128, ), (1, ))
    assert_size_stride(arg594_1, (128, ), (1, ))
    assert_size_stride(arg595_1, (), ())
    assert_size_stride(arg596_1, (256, ), (1, ))
    assert_size_stride(arg597_1, (256, ), (1, ))
    assert_size_stride(arg598_1, (), ())
    assert_size_stride(arg599_1, (64, ), (1, ))
    assert_size_stride(arg600_1, (64, ), (1, ))
    assert_size_stride(arg601_1, (), ())
    assert_size_stride(arg602_1, (512, ), (1, ))
    assert_size_stride(arg603_1, (512, ), (1, ))
    assert_size_stride(arg604_1, (), ())
    assert_size_stride(arg605_1, (128, ), (1, ))
    assert_size_stride(arg606_1, (128, ), (1, ))
    assert_size_stride(arg607_1, (), ())
    assert_size_stride(arg608_1, (256, ), (1, ))
    assert_size_stride(arg609_1, (256, ), (1, ))
    assert_size_stride(arg610_1, (), ())
    assert_size_stride(arg611_1, (64, ), (1, ))
    assert_size_stride(arg612_1, (64, ), (1, ))
    assert_size_stride(arg613_1, (), ())
    assert_size_stride(arg614_1, (512, ), (1, ))
    assert_size_stride(arg615_1, (512, ), (1, ))
    assert_size_stride(arg616_1, (), ())
    assert_size_stride(arg617_1, (256, ), (1, ))
    assert_size_stride(arg618_1, (256, ), (1, ))
    assert_size_stride(arg619_1, (), ())
    assert_size_stride(arg620_1, (512, ), (1, ))
    assert_size_stride(arg621_1, (512, ), (1, ))
    assert_size_stride(arg622_1, (), ())
    assert_size_stride(arg623_1, (128, ), (1, ))
    assert_size_stride(arg624_1, (128, ), (1, ))
    assert_size_stride(arg625_1, (), ())
    assert_size_stride(arg626_1, (1024, ), (1, ))
    assert_size_stride(arg627_1, (1024, ), (1, ))
    assert_size_stride(arg628_1, (), ())
    assert_size_stride(arg629_1, (1024, ), (1, ))
    assert_size_stride(arg630_1, (1024, ), (1, ))
    assert_size_stride(arg631_1, (), ())
    assert_size_stride(arg632_1, (256, ), (1, ))
    assert_size_stride(arg633_1, (256, ), (1, ))
    assert_size_stride(arg634_1, (), ())
    assert_size_stride(arg635_1, (512, ), (1, ))
    assert_size_stride(arg636_1, (512, ), (1, ))
    assert_size_stride(arg637_1, (), ())
    assert_size_stride(arg638_1, (128, ), (1, ))
    assert_size_stride(arg639_1, (128, ), (1, ))
    assert_size_stride(arg640_1, (), ())
    assert_size_stride(arg641_1, (1024, ), (1, ))
    assert_size_stride(arg642_1, (1024, ), (1, ))
    assert_size_stride(arg643_1, (), ())
    assert_size_stride(arg644_1, (256, ), (1, ))
    assert_size_stride(arg645_1, (256, ), (1, ))
    assert_size_stride(arg646_1, (), ())
    assert_size_stride(arg647_1, (512, ), (1, ))
    assert_size_stride(arg648_1, (512, ), (1, ))
    assert_size_stride(arg649_1, (), ())
    assert_size_stride(arg650_1, (128, ), (1, ))
    assert_size_stride(arg651_1, (128, ), (1, ))
    assert_size_stride(arg652_1, (), ())
    assert_size_stride(arg653_1, (1024, ), (1, ))
    assert_size_stride(arg654_1, (1024, ), (1, ))
    assert_size_stride(arg655_1, (), ())
    assert_size_stride(arg656_1, (256, ), (1, ))
    assert_size_stride(arg657_1, (256, ), (1, ))
    assert_size_stride(arg658_1, (), ())
    assert_size_stride(arg659_1, (512, ), (1, ))
    assert_size_stride(arg660_1, (512, ), (1, ))
    assert_size_stride(arg661_1, (), ())
    assert_size_stride(arg662_1, (128, ), (1, ))
    assert_size_stride(arg663_1, (128, ), (1, ))
    assert_size_stride(arg664_1, (), ())
    assert_size_stride(arg665_1, (1024, ), (1, ))
    assert_size_stride(arg666_1, (1024, ), (1, ))
    assert_size_stride(arg667_1, (), ())
    assert_size_stride(arg668_1, (256, ), (1, ))
    assert_size_stride(arg669_1, (256, ), (1, ))
    assert_size_stride(arg670_1, (), ())
    assert_size_stride(arg671_1, (512, ), (1, ))
    assert_size_stride(arg672_1, (512, ), (1, ))
    assert_size_stride(arg673_1, (), ())
    assert_size_stride(arg674_1, (128, ), (1, ))
    assert_size_stride(arg675_1, (128, ), (1, ))
    assert_size_stride(arg676_1, (), ())
    assert_size_stride(arg677_1, (1024, ), (1, ))
    assert_size_stride(arg678_1, (1024, ), (1, ))
    assert_size_stride(arg679_1, (), ())
    assert_size_stride(arg680_1, (256, ), (1, ))
    assert_size_stride(arg681_1, (256, ), (1, ))
    assert_size_stride(arg682_1, (), ())
    assert_size_stride(arg683_1, (512, ), (1, ))
    assert_size_stride(arg684_1, (512, ), (1, ))
    assert_size_stride(arg685_1, (), ())
    assert_size_stride(arg686_1, (128, ), (1, ))
    assert_size_stride(arg687_1, (128, ), (1, ))
    assert_size_stride(arg688_1, (), ())
    assert_size_stride(arg689_1, (1024, ), (1, ))
    assert_size_stride(arg690_1, (1024, ), (1, ))
    assert_size_stride(arg691_1, (), ())
    assert_size_stride(arg692_1, (256, ), (1, ))
    assert_size_stride(arg693_1, (256, ), (1, ))
    assert_size_stride(arg694_1, (), ())
    assert_size_stride(arg695_1, (512, ), (1, ))
    assert_size_stride(arg696_1, (512, ), (1, ))
    assert_size_stride(arg697_1, (), ())
    assert_size_stride(arg698_1, (128, ), (1, ))
    assert_size_stride(arg699_1, (128, ), (1, ))
    assert_size_stride(arg700_1, (), ())
    assert_size_stride(arg701_1, (1024, ), (1, ))
    assert_size_stride(arg702_1, (1024, ), (1, ))
    assert_size_stride(arg703_1, (), ())
    assert_size_stride(arg704_1, (256, ), (1, ))
    assert_size_stride(arg705_1, (256, ), (1, ))
    assert_size_stride(arg706_1, (), ())
    assert_size_stride(arg707_1, (512, ), (1, ))
    assert_size_stride(arg708_1, (512, ), (1, ))
    assert_size_stride(arg709_1, (), ())
    assert_size_stride(arg710_1, (128, ), (1, ))
    assert_size_stride(arg711_1, (128, ), (1, ))
    assert_size_stride(arg712_1, (), ())
    assert_size_stride(arg713_1, (1024, ), (1, ))
    assert_size_stride(arg714_1, (1024, ), (1, ))
    assert_size_stride(arg715_1, (), ())
    assert_size_stride(arg716_1, (256, ), (1, ))
    assert_size_stride(arg717_1, (256, ), (1, ))
    assert_size_stride(arg718_1, (), ())
    assert_size_stride(arg719_1, (512, ), (1, ))
    assert_size_stride(arg720_1, (512, ), (1, ))
    assert_size_stride(arg721_1, (), ())
    assert_size_stride(arg722_1, (128, ), (1, ))
    assert_size_stride(arg723_1, (128, ), (1, ))
    assert_size_stride(arg724_1, (), ())
    assert_size_stride(arg725_1, (1024, ), (1, ))
    assert_size_stride(arg726_1, (1024, ), (1, ))
    assert_size_stride(arg727_1, (), ())
    assert_size_stride(arg728_1, (256, ), (1, ))
    assert_size_stride(arg729_1, (256, ), (1, ))
    assert_size_stride(arg730_1, (), ())
    assert_size_stride(arg731_1, (512, ), (1, ))
    assert_size_stride(arg732_1, (512, ), (1, ))
    assert_size_stride(arg733_1, (), ())
    assert_size_stride(arg734_1, (128, ), (1, ))
    assert_size_stride(arg735_1, (128, ), (1, ))
    assert_size_stride(arg736_1, (), ())
    assert_size_stride(arg737_1, (1024, ), (1, ))
    assert_size_stride(arg738_1, (1024, ), (1, ))
    assert_size_stride(arg739_1, (), ())
    assert_size_stride(arg740_1, (256, ), (1, ))
    assert_size_stride(arg741_1, (256, ), (1, ))
    assert_size_stride(arg742_1, (), ())
    assert_size_stride(arg743_1, (512, ), (1, ))
    assert_size_stride(arg744_1, (512, ), (1, ))
    assert_size_stride(arg745_1, (), ())
    assert_size_stride(arg746_1, (128, ), (1, ))
    assert_size_stride(arg747_1, (128, ), (1, ))
    assert_size_stride(arg748_1, (), ())
    assert_size_stride(arg749_1, (1024, ), (1, ))
    assert_size_stride(arg750_1, (1024, ), (1, ))
    assert_size_stride(arg751_1, (), ())
    assert_size_stride(arg752_1, (256, ), (1, ))
    assert_size_stride(arg753_1, (256, ), (1, ))
    assert_size_stride(arg754_1, (), ())
    assert_size_stride(arg755_1, (512, ), (1, ))
    assert_size_stride(arg756_1, (512, ), (1, ))
    assert_size_stride(arg757_1, (), ())
    assert_size_stride(arg758_1, (128, ), (1, ))
    assert_size_stride(arg759_1, (128, ), (1, ))
    assert_size_stride(arg760_1, (), ())
    assert_size_stride(arg761_1, (1024, ), (1, ))
    assert_size_stride(arg762_1, (1024, ), (1, ))
    assert_size_stride(arg763_1, (), ())
    assert_size_stride(arg764_1, (256, ), (1, ))
    assert_size_stride(arg765_1, (256, ), (1, ))
    assert_size_stride(arg766_1, (), ())
    assert_size_stride(arg767_1, (512, ), (1, ))
    assert_size_stride(arg768_1, (512, ), (1, ))
    assert_size_stride(arg769_1, (), ())
    assert_size_stride(arg770_1, (128, ), (1, ))
    assert_size_stride(arg771_1, (128, ), (1, ))
    assert_size_stride(arg772_1, (), ())
    assert_size_stride(arg773_1, (1024, ), (1, ))
    assert_size_stride(arg774_1, (1024, ), (1, ))
    assert_size_stride(arg775_1, (), ())
    assert_size_stride(arg776_1, (256, ), (1, ))
    assert_size_stride(arg777_1, (256, ), (1, ))
    assert_size_stride(arg778_1, (), ())
    assert_size_stride(arg779_1, (512, ), (1, ))
    assert_size_stride(arg780_1, (512, ), (1, ))
    assert_size_stride(arg781_1, (), ())
    assert_size_stride(arg782_1, (128, ), (1, ))
    assert_size_stride(arg783_1, (128, ), (1, ))
    assert_size_stride(arg784_1, (), ())
    assert_size_stride(arg785_1, (1024, ), (1, ))
    assert_size_stride(arg786_1, (1024, ), (1, ))
    assert_size_stride(arg787_1, (), ())
    assert_size_stride(arg788_1, (256, ), (1, ))
    assert_size_stride(arg789_1, (256, ), (1, ))
    assert_size_stride(arg790_1, (), ())
    assert_size_stride(arg791_1, (512, ), (1, ))
    assert_size_stride(arg792_1, (512, ), (1, ))
    assert_size_stride(arg793_1, (), ())
    assert_size_stride(arg794_1, (128, ), (1, ))
    assert_size_stride(arg795_1, (128, ), (1, ))
    assert_size_stride(arg796_1, (), ())
    assert_size_stride(arg797_1, (1024, ), (1, ))
    assert_size_stride(arg798_1, (1024, ), (1, ))
    assert_size_stride(arg799_1, (), ())
    assert_size_stride(arg800_1, (256, ), (1, ))
    assert_size_stride(arg801_1, (256, ), (1, ))
    assert_size_stride(arg802_1, (), ())
    assert_size_stride(arg803_1, (512, ), (1, ))
    assert_size_stride(arg804_1, (512, ), (1, ))
    assert_size_stride(arg805_1, (), ())
    assert_size_stride(arg806_1, (128, ), (1, ))
    assert_size_stride(arg807_1, (128, ), (1, ))
    assert_size_stride(arg808_1, (), ())
    assert_size_stride(arg809_1, (1024, ), (1, ))
    assert_size_stride(arg810_1, (1024, ), (1, ))
    assert_size_stride(arg811_1, (), ())
    assert_size_stride(arg812_1, (256, ), (1, ))
    assert_size_stride(arg813_1, (256, ), (1, ))
    assert_size_stride(arg814_1, (), ())
    assert_size_stride(arg815_1, (512, ), (1, ))
    assert_size_stride(arg816_1, (512, ), (1, ))
    assert_size_stride(arg817_1, (), ())
    assert_size_stride(arg818_1, (128, ), (1, ))
    assert_size_stride(arg819_1, (128, ), (1, ))
    assert_size_stride(arg820_1, (), ())
    assert_size_stride(arg821_1, (1024, ), (1, ))
    assert_size_stride(arg822_1, (1024, ), (1, ))
    assert_size_stride(arg823_1, (), ())
    assert_size_stride(arg824_1, (256, ), (1, ))
    assert_size_stride(arg825_1, (256, ), (1, ))
    assert_size_stride(arg826_1, (), ())
    assert_size_stride(arg827_1, (512, ), (1, ))
    assert_size_stride(arg828_1, (512, ), (1, ))
    assert_size_stride(arg829_1, (), ())
    assert_size_stride(arg830_1, (128, ), (1, ))
    assert_size_stride(arg831_1, (128, ), (1, ))
    assert_size_stride(arg832_1, (), ())
    assert_size_stride(arg833_1, (1024, ), (1, ))
    assert_size_stride(arg834_1, (1024, ), (1, ))
    assert_size_stride(arg835_1, (), ())
    assert_size_stride(arg836_1, (256, ), (1, ))
    assert_size_stride(arg837_1, (256, ), (1, ))
    assert_size_stride(arg838_1, (), ())
    assert_size_stride(arg839_1, (512, ), (1, ))
    assert_size_stride(arg840_1, (512, ), (1, ))
    assert_size_stride(arg841_1, (), ())
    assert_size_stride(arg842_1, (128, ), (1, ))
    assert_size_stride(arg843_1, (128, ), (1, ))
    assert_size_stride(arg844_1, (), ())
    assert_size_stride(arg845_1, (1024, ), (1, ))
    assert_size_stride(arg846_1, (1024, ), (1, ))
    assert_size_stride(arg847_1, (), ())
    assert_size_stride(arg848_1, (256, ), (1, ))
    assert_size_stride(arg849_1, (256, ), (1, ))
    assert_size_stride(arg850_1, (), ())
    assert_size_stride(arg851_1, (512, ), (1, ))
    assert_size_stride(arg852_1, (512, ), (1, ))
    assert_size_stride(arg853_1, (), ())
    assert_size_stride(arg854_1, (128, ), (1, ))
    assert_size_stride(arg855_1, (128, ), (1, ))
    assert_size_stride(arg856_1, (), ())
    assert_size_stride(arg857_1, (1024, ), (1, ))
    assert_size_stride(arg858_1, (1024, ), (1, ))
    assert_size_stride(arg859_1, (), ())
    assert_size_stride(arg860_1, (256, ), (1, ))
    assert_size_stride(arg861_1, (256, ), (1, ))
    assert_size_stride(arg862_1, (), ())
    assert_size_stride(arg863_1, (512, ), (1, ))
    assert_size_stride(arg864_1, (512, ), (1, ))
    assert_size_stride(arg865_1, (), ())
    assert_size_stride(arg866_1, (128, ), (1, ))
    assert_size_stride(arg867_1, (128, ), (1, ))
    assert_size_stride(arg868_1, (), ())
    assert_size_stride(arg869_1, (1024, ), (1, ))
    assert_size_stride(arg870_1, (1024, ), (1, ))
    assert_size_stride(arg871_1, (), ())
    assert_size_stride(arg872_1, (256, ), (1, ))
    assert_size_stride(arg873_1, (256, ), (1, ))
    assert_size_stride(arg874_1, (), ())
    assert_size_stride(arg875_1, (512, ), (1, ))
    assert_size_stride(arg876_1, (512, ), (1, ))
    assert_size_stride(arg877_1, (), ())
    assert_size_stride(arg878_1, (128, ), (1, ))
    assert_size_stride(arg879_1, (128, ), (1, ))
    assert_size_stride(arg880_1, (), ())
    assert_size_stride(arg881_1, (1024, ), (1, ))
    assert_size_stride(arg882_1, (1024, ), (1, ))
    assert_size_stride(arg883_1, (), ())
    assert_size_stride(arg884_1, (256, ), (1, ))
    assert_size_stride(arg885_1, (256, ), (1, ))
    assert_size_stride(arg886_1, (), ())
    assert_size_stride(arg887_1, (512, ), (1, ))
    assert_size_stride(arg888_1, (512, ), (1, ))
    assert_size_stride(arg889_1, (), ())
    assert_size_stride(arg890_1, (128, ), (1, ))
    assert_size_stride(arg891_1, (128, ), (1, ))
    assert_size_stride(arg892_1, (), ())
    assert_size_stride(arg893_1, (1024, ), (1, ))
    assert_size_stride(arg894_1, (1024, ), (1, ))
    assert_size_stride(arg895_1, (), ())
    assert_size_stride(arg896_1, (512, ), (1, ))
    assert_size_stride(arg897_1, (512, ), (1, ))
    assert_size_stride(arg898_1, (), ())
    assert_size_stride(arg899_1, (1024, ), (1, ))
    assert_size_stride(arg900_1, (1024, ), (1, ))
    assert_size_stride(arg901_1, (), ())
    assert_size_stride(arg902_1, (256, ), (1, ))
    assert_size_stride(arg903_1, (256, ), (1, ))
    assert_size_stride(arg904_1, (), ())
    assert_size_stride(arg905_1, (2048, ), (1, ))
    assert_size_stride(arg906_1, (2048, ), (1, ))
    assert_size_stride(arg907_1, (), ())
    assert_size_stride(arg908_1, (2048, ), (1, ))
    assert_size_stride(arg909_1, (2048, ), (1, ))
    assert_size_stride(arg910_1, (), ())
    assert_size_stride(arg911_1, (512, ), (1, ))
    assert_size_stride(arg912_1, (512, ), (1, ))
    assert_size_stride(arg913_1, (), ())
    assert_size_stride(arg914_1, (1024, ), (1, ))
    assert_size_stride(arg915_1, (1024, ), (1, ))
    assert_size_stride(arg916_1, (), ())
    assert_size_stride(arg917_1, (256, ), (1, ))
    assert_size_stride(arg918_1, (256, ), (1, ))
    assert_size_stride(arg919_1, (), ())
    assert_size_stride(arg920_1, (2048, ), (1, ))
    assert_size_stride(arg921_1, (2048, ), (1, ))
    assert_size_stride(arg922_1, (), ())
    assert_size_stride(arg923_1, (512, ), (1, ))
    assert_size_stride(arg924_1, (512, ), (1, ))
    assert_size_stride(arg925_1, (), ())
    assert_size_stride(arg926_1, (1024, ), (1, ))
    assert_size_stride(arg927_1, (1024, ), (1, ))
    assert_size_stride(arg928_1, (), ())
    assert_size_stride(arg929_1, (256, ), (1, ))
    assert_size_stride(arg930_1, (256, ), (1, ))
    assert_size_stride(arg931_1, (), ())
    assert_size_stride(arg932_1, (2048, ), (1, ))
    assert_size_stride(arg933_1, (2048, ), (1, ))
    assert_size_stride(arg934_1, (), ())
    assert_size_stride(arg935_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    buf0 = empty_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg935_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg935_1
    # Source Nodes: [l__mod___conv1_0], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg518_1.data_ptr()), c_void_p(arg519_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg1_1
    del arg2_1
    del arg3_1
    del arg518_1
    del arg519_1
    # Source Nodes: [l__mod___conv1_1, l__mod___conv1_2, l__mod___conv1_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf5, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    del buf3
    buf6 = buf5; del buf5  # reuse
    buf7 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_2(c_void_p(buf6.data_ptr()), c_void_p(arg521_1.data_ptr()), c_void_p(arg522_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf7.data_ptr()))
    del arg4_1
    del arg521_1
    del arg522_1
    del arg5_1
    del arg6_1
    # Source Nodes: [l__mod___conv1_4, l__mod___conv1_5, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf8 = extern_kernels.convolution(buf6, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf8, (8, 128, 128, 128), (2097152, 1, 16384, 128))
    del buf6
    del buf7
    buf9 = buf8; del buf8  # reuse
    buf10 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3(c_void_p(buf9.data_ptr()), c_void_p(arg524_1.data_ptr()), c_void_p(arg525_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf10.data_ptr()))
    del arg524_1
    del arg525_1
    del arg7_1
    del arg8_1
    del buf9
    # Source Nodes: [out], Original ATen: [aten.convolution]
    buf11 = extern_kernels.convolution(buf10, arg9_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf11, (8, 64, 64, 64), (262144, 1, 4096, 64))
    del arg9_1
    buf12 = buf11; del buf11  # reuse
    buf13 = reinterpret_tensor(buf4, (128, 32, 3, 3), (288, 1, 96, 32), 0); del buf4  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_4(c_void_p(buf12.data_ptr()), c_void_p(arg527_1.data_ptr()), c_void_p(arg528_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(buf13.data_ptr()))
    del arg10_1
    del arg11_1
    del arg12_1
    del arg527_1
    del arg528_1
    # Source Nodes: [out_1, out_2, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf14 = extern_kernels.convolution(buf12, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf14, (8, 128, 64, 64), (524288, 1, 8192, 128))
    buf15 = buf14; del buf14  # reuse
    buf16 = empty_strided((8, 64, 1, 1), (64, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf17 = reinterpret_tensor(buf16, (8, 64, 1, 1), (64, 1, 64, 64), 0); del buf16  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_5(c_void_p(buf15.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(arg530_1.data_ptr()), c_void_p(arg531_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg14_1.data_ptr()))
    del arg13_1
    del arg14_1
    del arg530_1
    del arg531_1
    # Source Nodes: [x_gap, x_gap_1, x_gap_2], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf18 = extern_kernels.convolution(buf17, arg15_1, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf18, (8, 32, 1, 1), (32, 1, 32, 32))
    del arg15_1
    del arg16_1
    buf19 = buf18; del buf18  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_6(c_void_p(buf19.data_ptr()), c_void_p(arg533_1.data_ptr()), c_void_p(arg534_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(arg18_1.data_ptr()))
    del arg17_1
    del arg18_1
    del arg533_1
    del arg534_1
    # Source Nodes: [x_attn, x_gap_3, x_gap_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf20 = extern_kernels.convolution(buf19, arg19_1, arg20_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf20, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg19_1
    del arg20_1
    del buf19
    buf21 = empty_strided((8, 2, 1, 64), (128, 64, 1024, 1), device='cpu', dtype=torch.float32)
    buf22 = buf12; del buf12  # reuse
    cpp_fused__softmax_mul_sum_7(c_void_p(buf20.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    del buf15
    del buf20
    # Source Nodes: [mul, out_3, out_8], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf23 = extern_kernels.convolution(buf22, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf23, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    del arg21_1
    del buf22
    # Source Nodes: [getattr_l__mod___layer1___0___downsample_1], Original ATen: [aten.convolution]
    buf24 = extern_kernels.convolution(buf10, arg24_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf24, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    del arg24_1
    del buf10
    buf25 = buf23; del buf23  # reuse
    buf26 = buf25; del buf25  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_8(c_void_p(buf26.data_ptr()), c_void_p(arg536_1.data_ptr()), c_void_p(arg537_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(arg539_1.data_ptr()), c_void_p(arg540_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(arg26_1.data_ptr()))
    del arg22_1
    del arg23_1
    del arg25_1
    del arg26_1
    del arg536_1
    del arg537_1
    del arg539_1
    del arg540_1
    del buf24
    # Source Nodes: [out_12, shortcut_2], Original ATen: [aten.convolution, aten.relu]
    buf27 = extern_kernels.convolution(buf26, arg27_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf27, (8, 64, 64, 64), (262144, 1, 4096, 64))
    del arg27_1
    buf28 = buf27; del buf27  # reuse
    buf29 = buf13; del buf13  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_9(c_void_p(buf28.data_ptr()), c_void_p(arg542_1.data_ptr()), c_void_p(arg543_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(buf29.data_ptr()))
    del arg28_1
    del arg29_1
    del arg30_1
    del arg542_1
    del arg543_1
    # Source Nodes: [out_13, out_14, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf30 = extern_kernels.convolution(buf28, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf30, (8, 128, 64, 64), (524288, 1, 8192, 128))
    buf31 = buf30; del buf30  # reuse
    buf32 = reinterpret_tensor(buf17, (8, 64, 1, 1), (64, 1, 512, 512), 0); del buf17  # reuse
    buf33 = reinterpret_tensor(buf32, (8, 64, 1, 1), (64, 1, 64, 64), 0); del buf32  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_10(c_void_p(buf31.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(arg545_1.data_ptr()), c_void_p(arg546_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg32_1.data_ptr()))
    del arg31_1
    del arg32_1
    del arg545_1
    del arg546_1
    # Source Nodes: [x_gap_5, x_gap_6, x_gap_7], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf34 = extern_kernels.convolution(buf33, arg33_1, arg34_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf34, (8, 32, 1, 1), (32, 1, 32, 32))
    del arg33_1
    del arg34_1
    buf35 = buf34; del buf34  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_11(c_void_p(buf35.data_ptr()), c_void_p(arg548_1.data_ptr()), c_void_p(arg549_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(arg36_1.data_ptr()))
    del arg35_1
    del arg36_1
    del arg548_1
    del arg549_1
    # Source Nodes: [x_attn_2, x_gap_8, x_gap_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf36 = extern_kernels.convolution(buf35, arg37_1, arg38_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf36, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg37_1
    del arg38_1
    del buf35
    buf37 = buf21; del buf21  # reuse
    buf38 = buf28; del buf28  # reuse
    cpp_fused__softmax_mul_sum_12(c_void_p(buf36.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()))
    del buf31
    del buf36
    # Source Nodes: [mul_1, out_15, out_20], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf39 = extern_kernels.convolution(buf38, arg39_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf39, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    del arg39_1
    del buf38
    buf40 = buf26; del buf26  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_13(c_void_p(buf40.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(arg551_1.data_ptr()), c_void_p(arg552_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()))
    del arg40_1
    del arg41_1
    del arg551_1
    del arg552_1
    del buf39
    # Source Nodes: [out_24], Original ATen: [aten.convolution]
    buf41 = extern_kernels.convolution(buf40, arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf41, (8, 64, 64, 64), (262144, 1, 4096, 64))
    del arg42_1
    buf42 = buf41; del buf41  # reuse
    buf43 = buf29; del buf29  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_14(c_void_p(buf42.data_ptr()), c_void_p(arg554_1.data_ptr()), c_void_p(arg555_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(buf43.data_ptr()))
    del arg43_1
    del arg44_1
    del arg45_1
    del arg554_1
    del arg555_1
    # Source Nodes: [out_25, out_26, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf44 = extern_kernels.convolution(buf42, buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf44, (8, 128, 64, 64), (524288, 1, 8192, 128))
    del buf43
    buf45 = buf44; del buf44  # reuse
    buf46 = reinterpret_tensor(buf33, (8, 64, 1, 1), (64, 1, 512, 512), 0); del buf33  # reuse
    buf47 = reinterpret_tensor(buf46, (8, 64, 1, 1), (64, 1, 64, 64), 0); del buf46  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_15(c_void_p(buf45.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(arg557_1.data_ptr()), c_void_p(arg558_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()))
    del arg46_1
    del arg47_1
    del arg557_1
    del arg558_1
    # Source Nodes: [x_gap_10, x_gap_11, x_gap_12], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf48 = extern_kernels.convolution(buf47, arg48_1, arg49_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf48, (8, 32, 1, 1), (32, 1, 32, 32))
    del arg48_1
    del arg49_1
    del buf47
    buf49 = buf48; del buf48  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_16(c_void_p(buf49.data_ptr()), c_void_p(arg560_1.data_ptr()), c_void_p(arg561_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()))
    del arg50_1
    del arg51_1
    del arg560_1
    del arg561_1
    # Source Nodes: [x_attn_4, x_gap_13, x_gap_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf50 = extern_kernels.convolution(buf49, arg52_1, arg53_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf50, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg52_1
    del arg53_1
    del buf49
    buf51 = buf37; del buf37  # reuse
    buf52 = buf42; del buf42  # reuse
    cpp_fused__softmax_mul_sum_17(c_void_p(buf50.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()))
    del buf45
    del buf50
    # Source Nodes: [mul_2, out_27, out_32], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf53 = extern_kernels.convolution(buf52, arg54_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf53, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    del arg54_1
    buf54 = buf40; del buf40  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_18(c_void_p(buf54.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(arg563_1.data_ptr()), c_void_p(arg564_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(arg56_1.data_ptr()))
    del arg55_1
    del arg563_1
    del arg564_1
    del arg56_1
    del buf53
    # Source Nodes: [out_36], Original ATen: [aten.convolution]
    buf55 = extern_kernels.convolution(buf54, arg57_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf55, (8, 128, 64, 64), (524288, 1, 8192, 128))
    del arg57_1
    buf56 = buf55; del buf55  # reuse
    buf57 = empty_strided((256, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_19(c_void_p(buf56.data_ptr()), c_void_p(arg566_1.data_ptr()), c_void_p(arg567_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(buf57.data_ptr()))
    del arg566_1
    del arg567_1
    del arg58_1
    del arg59_1
    del arg60_1
    # Source Nodes: [out_37, out_38, x_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf58 = extern_kernels.convolution(buf56, buf57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf58, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    buf59 = buf58; del buf58  # reuse
    buf60 = reinterpret_tensor(buf51, (8, 128, 1, 1), (128, 1, 1024, 1024), 0); del buf51  # reuse
    buf61 = reinterpret_tensor(buf60, (8, 128, 1, 1), (128, 1, 128, 128), 0); del buf60  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_20(c_void_p(buf59.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(arg569_1.data_ptr()), c_void_p(arg570_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg62_1.data_ptr()))
    del arg569_1
    del arg570_1
    del arg61_1
    del arg62_1
    # Source Nodes: [x_gap_15, x_gap_16, x_gap_17], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf62 = extern_kernels.convolution(buf61, arg63_1, arg64_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf62, (8, 64, 1, 1), (64, 1, 64, 64))
    del arg63_1
    del arg64_1
    buf63 = buf62; del buf62  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_21(c_void_p(buf63.data_ptr()), c_void_p(arg572_1.data_ptr()), c_void_p(arg573_1.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(arg66_1.data_ptr()))
    del arg572_1
    del arg573_1
    del arg65_1
    del arg66_1
    # Source Nodes: [x_attn_6, x_gap_18, x_gap_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf64 = extern_kernels.convolution(buf63, arg67_1, arg68_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf64, (8, 256, 1, 1), (256, 1, 256, 256))
    del arg67_1
    del arg68_1
    del buf63
    buf65 = empty_strided((8, 2, 1, 128), (256, 128, 2048, 1), device='cpu', dtype=torch.float32)
    buf66 = buf56; del buf56  # reuse
    buf67 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_avg_pool2d_mul_sum_22(c_void_p(buf64.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()))
    del buf59
    del buf64
    del buf66
    # Source Nodes: [out_45], Original ATen: [aten.convolution]
    buf68 = extern_kernels.convolution(buf67, arg69_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf68, (8, 512, 32, 32), (524288, 1, 16384, 512))
    del arg69_1
    del buf67
    buf69 = reinterpret_tensor(buf52, (8, 256, 32, 32), (262144, 1, 8192, 256), 0); del buf52  # reuse
    cpp_fused_avg_pool2d_23(c_void_p(buf54.data_ptr()), c_void_p(buf69.data_ptr()))
    del buf54
    # Source Nodes: [getattr_l__mod___layer2___0___downsample_0, getattr_l__mod___layer2___0___downsample_1], Original ATen: [aten.avg_pool2d, aten.convolution]
    buf70 = extern_kernels.convolution(buf69, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf70, (8, 512, 32, 32), (524288, 1, 16384, 512))
    del arg72_1
    del buf69
    buf71 = buf68; del buf68  # reuse
    buf72 = buf71; del buf71  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_24(c_void_p(buf72.data_ptr()), c_void_p(arg575_1.data_ptr()), c_void_p(arg576_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(arg578_1.data_ptr()), c_void_p(arg579_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(arg74_1.data_ptr()))
    del arg575_1
    del arg576_1
    del arg578_1
    del arg579_1
    del arg70_1
    del arg71_1
    del arg73_1
    del arg74_1
    del buf70
    # Source Nodes: [out_49, shortcut_6], Original ATen: [aten.convolution, aten.relu]
    buf73 = extern_kernels.convolution(buf72, arg75_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf73, (8, 128, 32, 32), (131072, 1, 4096, 128))
    del arg75_1
    buf74 = buf73; del buf73  # reuse
    buf75 = buf57; del buf57  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_25(c_void_p(buf74.data_ptr()), c_void_p(arg581_1.data_ptr()), c_void_p(arg582_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(buf75.data_ptr()))
    del arg581_1
    del arg582_1
    del arg76_1
    del arg77_1
    del arg78_1
    # Source Nodes: [out_50, out_51, x_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf76 = extern_kernels.convolution(buf74, buf75, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf76, (8, 256, 32, 32), (262144, 1, 8192, 256))
    buf77 = buf76; del buf76  # reuse
    buf78 = reinterpret_tensor(buf61, (8, 128, 1, 1), (128, 1, 1024, 1024), 0); del buf61  # reuse
    buf79 = reinterpret_tensor(buf78, (8, 128, 1, 1), (128, 1, 128, 128), 0); del buf78  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_26(c_void_p(buf77.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(arg584_1.data_ptr()), c_void_p(arg585_1.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(arg80_1.data_ptr()))
    del arg584_1
    del arg585_1
    del arg79_1
    del arg80_1
    # Source Nodes: [x_gap_20, x_gap_21, x_gap_22], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf80 = extern_kernels.convolution(buf79, arg81_1, arg82_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf80, (8, 64, 1, 1), (64, 1, 64, 64))
    del arg81_1
    del arg82_1
    buf81 = buf80; del buf80  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_27(c_void_p(buf81.data_ptr()), c_void_p(arg587_1.data_ptr()), c_void_p(arg588_1.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg84_1.data_ptr()))
    del arg587_1
    del arg588_1
    del arg83_1
    del arg84_1
    # Source Nodes: [x_attn_8, x_gap_23, x_gap_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf82 = extern_kernels.convolution(buf81, arg85_1, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf82, (8, 256, 1, 1), (256, 1, 256, 256))
    del arg85_1
    del arg86_1
    del buf81
    buf83 = buf65; del buf65  # reuse
    buf84 = buf74; del buf74  # reuse
    cpp_fused__softmax_mul_sum_28(c_void_p(buf82.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()))
    del buf77
    del buf82
    # Source Nodes: [mul_4, out_52, out_57], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf85 = extern_kernels.convolution(buf84, arg87_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf85, (8, 512, 32, 32), (524288, 1, 16384, 512))
    del arg87_1
    del buf84
    buf86 = buf72; del buf72  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_29(c_void_p(buf86.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(arg590_1.data_ptr()), c_void_p(arg591_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()))
    del arg590_1
    del arg591_1
    del arg88_1
    del arg89_1
    del buf85
    # Source Nodes: [out_61], Original ATen: [aten.convolution]
    buf87 = extern_kernels.convolution(buf86, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf87, (8, 128, 32, 32), (131072, 1, 4096, 128))
    del arg90_1
    buf88 = buf87; del buf87  # reuse
    buf89 = buf75; del buf75  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_30(c_void_p(buf88.data_ptr()), c_void_p(arg593_1.data_ptr()), c_void_p(arg594_1.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(buf89.data_ptr()))
    del arg593_1
    del arg594_1
    del arg91_1
    del arg92_1
    del arg93_1
    # Source Nodes: [out_62, out_63, x_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf90 = extern_kernels.convolution(buf88, buf89, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf90, (8, 256, 32, 32), (262144, 1, 8192, 256))
    buf91 = buf90; del buf90  # reuse
    buf92 = reinterpret_tensor(buf79, (8, 128, 1, 1), (128, 1, 1024, 1024), 0); del buf79  # reuse
    buf93 = reinterpret_tensor(buf92, (8, 128, 1, 1), (128, 1, 128, 128), 0); del buf92  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_31(c_void_p(buf91.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(arg596_1.data_ptr()), c_void_p(arg597_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()))
    del arg596_1
    del arg597_1
    del arg94_1
    del arg95_1
    # Source Nodes: [x_gap_25, x_gap_26, x_gap_27], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf94 = extern_kernels.convolution(buf93, arg96_1, arg97_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf94, (8, 64, 1, 1), (64, 1, 64, 64))
    del arg96_1
    del arg97_1
    buf95 = buf94; del buf94  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_32(c_void_p(buf95.data_ptr()), c_void_p(arg599_1.data_ptr()), c_void_p(arg600_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()))
    del arg599_1
    del arg600_1
    del arg98_1
    del arg99_1
    # Source Nodes: [x_attn_10, x_gap_28, x_gap_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf96 = extern_kernels.convolution(buf95, arg100_1, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf96, (8, 256, 1, 1), (256, 1, 256, 256))
    del arg100_1
    del arg101_1
    del buf95
    buf97 = buf83; del buf83  # reuse
    buf98 = buf88; del buf88  # reuse
    cpp_fused__softmax_mul_sum_33(c_void_p(buf96.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()))
    del buf91
    del buf96
    # Source Nodes: [mul_5, out_64, out_69], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf99 = extern_kernels.convolution(buf98, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf99, (8, 512, 32, 32), (524288, 1, 16384, 512))
    del arg102_1
    del buf98
    buf100 = buf86; del buf86  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_34(c_void_p(buf100.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(arg602_1.data_ptr()), c_void_p(arg603_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(arg104_1.data_ptr()))
    del arg103_1
    del arg104_1
    del arg602_1
    del arg603_1
    del buf99
    # Source Nodes: [out_73], Original ATen: [aten.convolution]
    buf101 = extern_kernels.convolution(buf100, arg105_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf101, (8, 128, 32, 32), (131072, 1, 4096, 128))
    del arg105_1
    buf102 = buf101; del buf101  # reuse
    buf103 = buf89; del buf89  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_35(c_void_p(buf102.data_ptr()), c_void_p(arg605_1.data_ptr()), c_void_p(arg606_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(buf103.data_ptr()))
    del arg106_1
    del arg107_1
    del arg108_1
    del arg605_1
    del arg606_1
    # Source Nodes: [out_74, out_75, x_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf104 = extern_kernels.convolution(buf102, buf103, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf104, (8, 256, 32, 32), (262144, 1, 8192, 256))
    del buf103
    buf105 = buf104; del buf104  # reuse
    buf106 = reinterpret_tensor(buf93, (8, 128, 1, 1), (128, 1, 1024, 1024), 0); del buf93  # reuse
    buf107 = reinterpret_tensor(buf106, (8, 128, 1, 1), (128, 1, 128, 128), 0); del buf106  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_36(c_void_p(buf105.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(arg608_1.data_ptr()), c_void_p(arg609_1.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(arg110_1.data_ptr()))
    del arg109_1
    del arg110_1
    del arg608_1
    del arg609_1
    # Source Nodes: [x_gap_30, x_gap_31, x_gap_32], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf108 = extern_kernels.convolution(buf107, arg111_1, arg112_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf108, (8, 64, 1, 1), (64, 1, 64, 64))
    del arg111_1
    del arg112_1
    del buf107
    buf109 = buf108; del buf108  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_37(c_void_p(buf109.data_ptr()), c_void_p(arg611_1.data_ptr()), c_void_p(arg612_1.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(arg114_1.data_ptr()))
    del arg113_1
    del arg114_1
    del arg611_1
    del arg612_1
    # Source Nodes: [x_attn_12, x_gap_33, x_gap_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf110 = extern_kernels.convolution(buf109, arg115_1, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf110, (8, 256, 1, 1), (256, 1, 256, 256))
    del arg115_1
    del arg116_1
    del buf109
    buf111 = buf97; del buf97  # reuse
    buf112 = buf102; del buf102  # reuse
    cpp_fused__softmax_mul_sum_38(c_void_p(buf110.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()))
    del buf105
    del buf110
    # Source Nodes: [mul_6, out_76, out_81], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf113 = extern_kernels.convolution(buf112, arg117_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf113, (8, 512, 32, 32), (524288, 1, 16384, 512))
    del arg117_1
    buf114 = buf100; del buf100  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_39(c_void_p(buf114.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(arg614_1.data_ptr()), c_void_p(arg615_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg119_1.data_ptr()))
    del arg118_1
    del arg119_1
    del arg614_1
    del arg615_1
    del buf113
    # Source Nodes: [out_85], Original ATen: [aten.convolution]
    buf115 = extern_kernels.convolution(buf114, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf115, (8, 256, 32, 32), (262144, 1, 8192, 256))
    del arg120_1
    buf116 = buf115; del buf115  # reuse
    buf117 = empty_strided((512, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_40(c_void_p(buf116.data_ptr()), c_void_p(arg617_1.data_ptr()), c_void_p(arg618_1.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(buf117.data_ptr()))
    del arg121_1
    del arg122_1
    del arg123_1
    del arg617_1
    del arg618_1
    # Source Nodes: [out_86, out_87, x_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf118 = extern_kernels.convolution(buf116, buf117, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf118, (8, 512, 32, 32), (524288, 1, 16384, 512))
    buf119 = buf118; del buf118  # reuse
    buf120 = reinterpret_tensor(buf111, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf111  # reuse
    buf121 = reinterpret_tensor(buf120, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf120  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_41(c_void_p(buf119.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(arg620_1.data_ptr()), c_void_p(arg621_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg125_1.data_ptr()))
    del arg124_1
    del arg125_1
    del arg620_1
    del arg621_1
    # Source Nodes: [x_gap_35, x_gap_36, x_gap_37], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf122 = extern_kernels.convolution(buf121, arg126_1, arg127_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf122, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg126_1
    del arg127_1
    buf123 = buf122; del buf122  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_42(c_void_p(buf123.data_ptr()), c_void_p(arg623_1.data_ptr()), c_void_p(arg624_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(arg129_1.data_ptr()))
    del arg128_1
    del arg129_1
    del arg623_1
    del arg624_1
    # Source Nodes: [x_attn_14, x_gap_38, x_gap_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf124 = extern_kernels.convolution(buf123, arg130_1, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf124, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg130_1
    del arg131_1
    del buf123
    buf125 = empty_strided((8, 2, 1, 256), (512, 256, 4096, 1), device='cpu', dtype=torch.float32)
    buf126 = buf116; del buf116  # reuse
    buf127 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_avg_pool2d_mul_sum_43(c_void_p(buf124.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()))
    del buf119
    del buf124
    del buf126
    # Source Nodes: [out_94], Original ATen: [aten.convolution]
    buf128 = extern_kernels.convolution(buf127, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf128, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg132_1
    del buf127
    buf129 = reinterpret_tensor(buf112, (8, 512, 16, 16), (131072, 1, 8192, 512), 0); del buf112  # reuse
    cpp_fused_avg_pool2d_44(c_void_p(buf114.data_ptr()), c_void_p(buf129.data_ptr()))
    del buf114
    # Source Nodes: [getattr_l__mod___layer3___0___downsample_0, getattr_l__mod___layer3___0___downsample_1], Original ATen: [aten.avg_pool2d, aten.convolution]
    buf130 = extern_kernels.convolution(buf129, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf130, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg135_1
    del buf129
    buf131 = buf128; del buf128  # reuse
    buf132 = buf131; del buf131  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_45(c_void_p(buf132.data_ptr()), c_void_p(arg626_1.data_ptr()), c_void_p(arg627_1.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(arg629_1.data_ptr()), c_void_p(arg630_1.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(arg137_1.data_ptr()))
    del arg133_1
    del arg134_1
    del arg136_1
    del arg137_1
    del arg626_1
    del arg627_1
    del arg629_1
    del arg630_1
    del buf130
    # Source Nodes: [out_98, shortcut_11], Original ATen: [aten.convolution, aten.relu]
    buf133 = extern_kernels.convolution(buf132, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf133, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg138_1
    buf134 = buf133; del buf133  # reuse
    buf135 = buf117; del buf117  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_46(c_void_p(buf134.data_ptr()), c_void_p(arg632_1.data_ptr()), c_void_p(arg633_1.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(buf135.data_ptr()))
    del arg139_1
    del arg140_1
    del arg141_1
    del arg632_1
    del arg633_1
    # Source Nodes: [out_100, out_99, x_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf136 = extern_kernels.convolution(buf134, buf135, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf136, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf137 = buf136; del buf136  # reuse
    buf138 = reinterpret_tensor(buf121, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf121  # reuse
    buf139 = reinterpret_tensor(buf138, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf138  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_47(c_void_p(buf137.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(arg635_1.data_ptr()), c_void_p(arg636_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg143_1.data_ptr()))
    del arg142_1
    del arg143_1
    del arg635_1
    del arg636_1
    # Source Nodes: [x_gap_40, x_gap_41, x_gap_42], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf140 = extern_kernels.convolution(buf139, arg144_1, arg145_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf140, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg144_1
    del arg145_1
    buf141 = buf140; del buf140  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_48(c_void_p(buf141.data_ptr()), c_void_p(arg638_1.data_ptr()), c_void_p(arg639_1.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(arg147_1.data_ptr()))
    del arg146_1
    del arg147_1
    del arg638_1
    del arg639_1
    # Source Nodes: [x_attn_16, x_gap_43, x_gap_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf142 = extern_kernels.convolution(buf141, arg148_1, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf142, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg148_1
    del arg149_1
    del buf141
    buf143 = buf125; del buf125  # reuse
    buf144 = buf134; del buf134  # reuse
    cpp_fused__softmax_mul_sum_49(c_void_p(buf142.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()))
    del buf137
    del buf142
    # Source Nodes: [mul_8, out_101, out_106], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf145 = extern_kernels.convolution(buf144, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf145, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg150_1
    del buf144
    buf146 = buf132; del buf132  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_50(c_void_p(buf146.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(arg641_1.data_ptr()), c_void_p(arg642_1.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(arg152_1.data_ptr()))
    del arg151_1
    del arg152_1
    del arg641_1
    del arg642_1
    del buf145
    # Source Nodes: [out_110], Original ATen: [aten.convolution]
    buf147 = extern_kernels.convolution(buf146, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf147, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg153_1
    buf148 = buf147; del buf147  # reuse
    buf149 = buf135; del buf135  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_51(c_void_p(buf148.data_ptr()), c_void_p(arg644_1.data_ptr()), c_void_p(arg645_1.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(arg155_1.data_ptr()), c_void_p(arg156_1.data_ptr()), c_void_p(buf149.data_ptr()))
    del arg154_1
    del arg155_1
    del arg156_1
    del arg644_1
    del arg645_1
    # Source Nodes: [out_111, out_112, x_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf150 = extern_kernels.convolution(buf148, buf149, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf150, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf151 = buf150; del buf150  # reuse
    buf152 = reinterpret_tensor(buf139, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf139  # reuse
    buf153 = reinterpret_tensor(buf152, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf152  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_52(c_void_p(buf151.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(arg647_1.data_ptr()), c_void_p(arg648_1.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(arg158_1.data_ptr()))
    del arg157_1
    del arg158_1
    del arg647_1
    del arg648_1
    # Source Nodes: [x_gap_45, x_gap_46, x_gap_47], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf154 = extern_kernels.convolution(buf153, arg159_1, arg160_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf154, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg159_1
    del arg160_1
    buf155 = buf154; del buf154  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_53(c_void_p(buf155.data_ptr()), c_void_p(arg650_1.data_ptr()), c_void_p(arg651_1.data_ptr()), c_void_p(arg161_1.data_ptr()), c_void_p(arg162_1.data_ptr()))
    del arg161_1
    del arg162_1
    del arg650_1
    del arg651_1
    # Source Nodes: [x_attn_18, x_gap_48, x_gap_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf156 = extern_kernels.convolution(buf155, arg163_1, arg164_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf156, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg163_1
    del arg164_1
    del buf155
    buf157 = buf143; del buf143  # reuse
    buf158 = buf148; del buf148  # reuse
    cpp_fused__softmax_mul_sum_54(c_void_p(buf156.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()))
    del buf151
    del buf156
    # Source Nodes: [mul_9, out_113, out_118], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf159 = extern_kernels.convolution(buf158, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf159, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg165_1
    del buf158
    buf160 = buf146; del buf146  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_55(c_void_p(buf160.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(arg653_1.data_ptr()), c_void_p(arg654_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(arg167_1.data_ptr()))
    del arg166_1
    del arg167_1
    del arg653_1
    del arg654_1
    del buf159
    # Source Nodes: [out_122], Original ATen: [aten.convolution]
    buf161 = extern_kernels.convolution(buf160, arg168_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf161, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg168_1
    buf162 = buf161; del buf161  # reuse
    buf163 = buf149; del buf149  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_56(c_void_p(buf162.data_ptr()), c_void_p(arg656_1.data_ptr()), c_void_p(arg657_1.data_ptr()), c_void_p(arg169_1.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(arg171_1.data_ptr()), c_void_p(buf163.data_ptr()))
    del arg169_1
    del arg170_1
    del arg171_1
    del arg656_1
    del arg657_1
    # Source Nodes: [out_123, out_124, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf164 = extern_kernels.convolution(buf162, buf163, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf164, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf165 = buf164; del buf164  # reuse
    buf166 = reinterpret_tensor(buf153, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf153  # reuse
    buf167 = reinterpret_tensor(buf166, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf166  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_57(c_void_p(buf165.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(arg659_1.data_ptr()), c_void_p(arg660_1.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(arg173_1.data_ptr()))
    del arg172_1
    del arg173_1
    del arg659_1
    del arg660_1
    # Source Nodes: [x_gap_50, x_gap_51, x_gap_52], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf168 = extern_kernels.convolution(buf167, arg174_1, arg175_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf168, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg174_1
    del arg175_1
    buf169 = buf168; del buf168  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_58(c_void_p(buf169.data_ptr()), c_void_p(arg662_1.data_ptr()), c_void_p(arg663_1.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(arg177_1.data_ptr()))
    del arg176_1
    del arg177_1
    del arg662_1
    del arg663_1
    # Source Nodes: [x_attn_20, x_gap_53, x_gap_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf170 = extern_kernels.convolution(buf169, arg178_1, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf170, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg178_1
    del arg179_1
    del buf169
    buf171 = buf157; del buf157  # reuse
    buf172 = buf162; del buf162  # reuse
    cpp_fused__softmax_mul_sum_59(c_void_p(buf170.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()))
    del buf165
    del buf170
    # Source Nodes: [mul_10, out_125, out_130], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf173 = extern_kernels.convolution(buf172, arg180_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf173, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg180_1
    del buf172
    buf174 = buf160; del buf160  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_60(c_void_p(buf174.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(arg665_1.data_ptr()), c_void_p(arg666_1.data_ptr()), c_void_p(arg181_1.data_ptr()), c_void_p(arg182_1.data_ptr()))
    del arg181_1
    del arg182_1
    del arg665_1
    del arg666_1
    del buf173
    # Source Nodes: [out_134], Original ATen: [aten.convolution]
    buf175 = extern_kernels.convolution(buf174, arg183_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf175, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg183_1
    buf176 = buf175; del buf175  # reuse
    buf177 = buf163; del buf163  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_61(c_void_p(buf176.data_ptr()), c_void_p(arg668_1.data_ptr()), c_void_p(arg669_1.data_ptr()), c_void_p(arg184_1.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(buf177.data_ptr()))
    del arg184_1
    del arg185_1
    del arg186_1
    del arg668_1
    del arg669_1
    # Source Nodes: [out_135, out_136, x_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf178 = extern_kernels.convolution(buf176, buf177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf178, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf179 = buf178; del buf178  # reuse
    buf180 = reinterpret_tensor(buf167, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf167  # reuse
    buf181 = reinterpret_tensor(buf180, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf180  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_62(c_void_p(buf179.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(arg671_1.data_ptr()), c_void_p(arg672_1.data_ptr()), c_void_p(arg187_1.data_ptr()), c_void_p(arg188_1.data_ptr()))
    del arg187_1
    del arg188_1
    del arg671_1
    del arg672_1
    # Source Nodes: [x_gap_55, x_gap_56, x_gap_57], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf182 = extern_kernels.convolution(buf181, arg189_1, arg190_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf182, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg189_1
    del arg190_1
    buf183 = buf182; del buf182  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_63(c_void_p(buf183.data_ptr()), c_void_p(arg674_1.data_ptr()), c_void_p(arg675_1.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(arg192_1.data_ptr()))
    del arg191_1
    del arg192_1
    del arg674_1
    del arg675_1
    # Source Nodes: [x_attn_22, x_gap_58, x_gap_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf184 = extern_kernels.convolution(buf183, arg193_1, arg194_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf184, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg193_1
    del arg194_1
    del buf183
    buf185 = buf171; del buf171  # reuse
    buf186 = buf176; del buf176  # reuse
    cpp_fused__softmax_mul_sum_64(c_void_p(buf184.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()))
    del buf179
    del buf184
    # Source Nodes: [mul_11, out_137, out_142], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf187 = extern_kernels.convolution(buf186, arg195_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf187, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg195_1
    del buf186
    buf188 = buf174; del buf174  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_65(c_void_p(buf188.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(arg677_1.data_ptr()), c_void_p(arg678_1.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(arg197_1.data_ptr()))
    del arg196_1
    del arg197_1
    del arg677_1
    del arg678_1
    del buf187
    # Source Nodes: [out_146], Original ATen: [aten.convolution]
    buf189 = extern_kernels.convolution(buf188, arg198_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf189, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg198_1
    buf190 = buf189; del buf189  # reuse
    buf191 = buf177; del buf177  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_66(c_void_p(buf190.data_ptr()), c_void_p(arg680_1.data_ptr()), c_void_p(arg681_1.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(buf191.data_ptr()))
    del arg199_1
    del arg200_1
    del arg201_1
    del arg680_1
    del arg681_1
    # Source Nodes: [out_147, out_148, x_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf192 = extern_kernels.convolution(buf190, buf191, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf192, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf193 = buf192; del buf192  # reuse
    buf194 = reinterpret_tensor(buf181, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf181  # reuse
    buf195 = reinterpret_tensor(buf194, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf194  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_67(c_void_p(buf193.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(arg683_1.data_ptr()), c_void_p(arg684_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg203_1.data_ptr()))
    del arg202_1
    del arg203_1
    del arg683_1
    del arg684_1
    # Source Nodes: [x_gap_60, x_gap_61, x_gap_62], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf196 = extern_kernels.convolution(buf195, arg204_1, arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf196, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg204_1
    del arg205_1
    buf197 = buf196; del buf196  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_68(c_void_p(buf197.data_ptr()), c_void_p(arg686_1.data_ptr()), c_void_p(arg687_1.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(arg207_1.data_ptr()))
    del arg206_1
    del arg207_1
    del arg686_1
    del arg687_1
    # Source Nodes: [x_attn_24, x_gap_63, x_gap_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf198 = extern_kernels.convolution(buf197, arg208_1, arg209_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf198, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg208_1
    del arg209_1
    del buf197
    buf199 = buf185; del buf185  # reuse
    buf200 = buf190; del buf190  # reuse
    cpp_fused__softmax_mul_sum_69(c_void_p(buf198.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()))
    del buf193
    del buf198
    # Source Nodes: [mul_12, out_149, out_154], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf201 = extern_kernels.convolution(buf200, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf201, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg210_1
    del buf200
    buf202 = buf188; del buf188  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_70(c_void_p(buf202.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(arg689_1.data_ptr()), c_void_p(arg690_1.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg212_1.data_ptr()))
    del arg211_1
    del arg212_1
    del arg689_1
    del arg690_1
    del buf201
    # Source Nodes: [out_158], Original ATen: [aten.convolution]
    buf203 = extern_kernels.convolution(buf202, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf203, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg213_1
    buf204 = buf203; del buf203  # reuse
    buf205 = buf191; del buf191  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_71(c_void_p(buf204.data_ptr()), c_void_p(arg692_1.data_ptr()), c_void_p(arg693_1.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(arg215_1.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(buf205.data_ptr()))
    del arg214_1
    del arg215_1
    del arg216_1
    del arg692_1
    del arg693_1
    # Source Nodes: [out_159, out_160, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf206 = extern_kernels.convolution(buf204, buf205, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf206, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf207 = buf206; del buf206  # reuse
    buf208 = reinterpret_tensor(buf195, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf195  # reuse
    buf209 = reinterpret_tensor(buf208, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf208  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_72(c_void_p(buf207.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(arg695_1.data_ptr()), c_void_p(arg696_1.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg218_1.data_ptr()))
    del arg217_1
    del arg218_1
    del arg695_1
    del arg696_1
    # Source Nodes: [x_gap_65, x_gap_66, x_gap_67], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf210 = extern_kernels.convolution(buf209, arg219_1, arg220_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf210, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg219_1
    del arg220_1
    buf211 = buf210; del buf210  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_73(c_void_p(buf211.data_ptr()), c_void_p(arg698_1.data_ptr()), c_void_p(arg699_1.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(arg222_1.data_ptr()))
    del arg221_1
    del arg222_1
    del arg698_1
    del arg699_1
    # Source Nodes: [x_attn_26, x_gap_68, x_gap_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf212 = extern_kernels.convolution(buf211, arg223_1, arg224_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf212, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg223_1
    del arg224_1
    del buf211
    buf213 = buf199; del buf199  # reuse
    buf214 = buf204; del buf204  # reuse
    cpp_fused__softmax_mul_sum_74(c_void_p(buf212.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()))
    del buf207
    del buf212
    # Source Nodes: [mul_13, out_161, out_166], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf215 = extern_kernels.convolution(buf214, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf215, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg225_1
    del buf214
    buf216 = buf202; del buf202  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_75(c_void_p(buf216.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(arg701_1.data_ptr()), c_void_p(arg702_1.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg227_1.data_ptr()))
    del arg226_1
    del arg227_1
    del arg701_1
    del arg702_1
    del buf215
    # Source Nodes: [out_170], Original ATen: [aten.convolution]
    buf217 = extern_kernels.convolution(buf216, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf217, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg228_1
    buf218 = buf217; del buf217  # reuse
    buf219 = buf205; del buf205  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_76(c_void_p(buf218.data_ptr()), c_void_p(arg704_1.data_ptr()), c_void_p(arg705_1.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(buf219.data_ptr()))
    del arg229_1
    del arg230_1
    del arg231_1
    del arg704_1
    del arg705_1
    # Source Nodes: [out_171, out_172, x_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf220 = extern_kernels.convolution(buf218, buf219, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf220, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf221 = buf220; del buf220  # reuse
    buf222 = reinterpret_tensor(buf209, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf209  # reuse
    buf223 = reinterpret_tensor(buf222, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf222  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_77(c_void_p(buf221.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(arg707_1.data_ptr()), c_void_p(arg708_1.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(arg233_1.data_ptr()))
    del arg232_1
    del arg233_1
    del arg707_1
    del arg708_1
    # Source Nodes: [x_gap_70, x_gap_71, x_gap_72], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf224 = extern_kernels.convolution(buf223, arg234_1, arg235_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf224, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg234_1
    del arg235_1
    buf225 = buf224; del buf224  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_78(c_void_p(buf225.data_ptr()), c_void_p(arg710_1.data_ptr()), c_void_p(arg711_1.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(arg237_1.data_ptr()))
    del arg236_1
    del arg237_1
    del arg710_1
    del arg711_1
    # Source Nodes: [x_attn_28, x_gap_73, x_gap_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf226 = extern_kernels.convolution(buf225, arg238_1, arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf226, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg238_1
    del arg239_1
    del buf225
    buf227 = buf213; del buf213  # reuse
    buf228 = buf218; del buf218  # reuse
    cpp_fused__softmax_mul_sum_79(c_void_p(buf226.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()))
    del buf221
    del buf226
    # Source Nodes: [mul_14, out_173, out_178], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf229 = extern_kernels.convolution(buf228, arg240_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf229, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg240_1
    del buf228
    buf230 = buf216; del buf216  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_80(c_void_p(buf230.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(arg713_1.data_ptr()), c_void_p(arg714_1.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(arg242_1.data_ptr()))
    del arg241_1
    del arg242_1
    del arg713_1
    del arg714_1
    del buf229
    # Source Nodes: [out_182], Original ATen: [aten.convolution]
    buf231 = extern_kernels.convolution(buf230, arg243_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf231, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg243_1
    buf232 = buf231; del buf231  # reuse
    buf233 = buf219; del buf219  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_81(c_void_p(buf232.data_ptr()), c_void_p(arg716_1.data_ptr()), c_void_p(arg717_1.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(buf233.data_ptr()))
    del arg244_1
    del arg245_1
    del arg246_1
    del arg716_1
    del arg717_1
    # Source Nodes: [out_183, out_184, x_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf234 = extern_kernels.convolution(buf232, buf233, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf234, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf235 = buf234; del buf234  # reuse
    buf236 = reinterpret_tensor(buf223, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf223  # reuse
    buf237 = reinterpret_tensor(buf236, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf236  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_82(c_void_p(buf235.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(arg719_1.data_ptr()), c_void_p(arg720_1.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(arg248_1.data_ptr()))
    del arg247_1
    del arg248_1
    del arg719_1
    del arg720_1
    # Source Nodes: [x_gap_75, x_gap_76, x_gap_77], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf238 = extern_kernels.convolution(buf237, arg249_1, arg250_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf238, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg249_1
    del arg250_1
    buf239 = buf238; del buf238  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_83(c_void_p(buf239.data_ptr()), c_void_p(arg722_1.data_ptr()), c_void_p(arg723_1.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(arg252_1.data_ptr()))
    del arg251_1
    del arg252_1
    del arg722_1
    del arg723_1
    # Source Nodes: [x_attn_30, x_gap_78, x_gap_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf240 = extern_kernels.convolution(buf239, arg253_1, arg254_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf240, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg253_1
    del arg254_1
    del buf239
    buf241 = buf227; del buf227  # reuse
    buf242 = buf232; del buf232  # reuse
    cpp_fused__softmax_mul_sum_84(c_void_p(buf240.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()))
    del buf235
    del buf240
    # Source Nodes: [mul_15, out_185, out_190], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf243 = extern_kernels.convolution(buf242, arg255_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf243, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg255_1
    del buf242
    buf244 = buf230; del buf230  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_85(c_void_p(buf244.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(arg725_1.data_ptr()), c_void_p(arg726_1.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(arg257_1.data_ptr()))
    del arg256_1
    del arg257_1
    del arg725_1
    del arg726_1
    del buf243
    # Source Nodes: [out_194], Original ATen: [aten.convolution]
    buf245 = extern_kernels.convolution(buf244, arg258_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf245, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg258_1
    buf246 = buf245; del buf245  # reuse
    buf247 = buf233; del buf233  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_86(c_void_p(buf246.data_ptr()), c_void_p(arg728_1.data_ptr()), c_void_p(arg729_1.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(arg260_1.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(buf247.data_ptr()))
    del arg259_1
    del arg260_1
    del arg261_1
    del arg728_1
    del arg729_1
    # Source Nodes: [out_195, out_196, x_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf248 = extern_kernels.convolution(buf246, buf247, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf248, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf249 = buf248; del buf248  # reuse
    buf250 = reinterpret_tensor(buf237, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf237  # reuse
    buf251 = reinterpret_tensor(buf250, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf250  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_87(c_void_p(buf249.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(arg731_1.data_ptr()), c_void_p(arg732_1.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg263_1.data_ptr()))
    del arg262_1
    del arg263_1
    del arg731_1
    del arg732_1
    # Source Nodes: [x_gap_80, x_gap_81, x_gap_82], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf252 = extern_kernels.convolution(buf251, arg264_1, arg265_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf252, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg264_1
    del arg265_1
    buf253 = buf252; del buf252  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_88(c_void_p(buf253.data_ptr()), c_void_p(arg734_1.data_ptr()), c_void_p(arg735_1.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(arg267_1.data_ptr()))
    del arg266_1
    del arg267_1
    del arg734_1
    del arg735_1
    # Source Nodes: [x_attn_32, x_gap_83, x_gap_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf254 = extern_kernels.convolution(buf253, arg268_1, arg269_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf254, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg268_1
    del arg269_1
    del buf253
    buf255 = buf241; del buf241  # reuse
    buf256 = buf246; del buf246  # reuse
    cpp_fused__softmax_mul_sum_89(c_void_p(buf254.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()))
    del buf249
    del buf254
    # Source Nodes: [mul_16, out_197, out_202], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf257 = extern_kernels.convolution(buf256, arg270_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf257, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg270_1
    del buf256
    buf258 = buf244; del buf244  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_90(c_void_p(buf258.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(arg737_1.data_ptr()), c_void_p(arg738_1.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg272_1.data_ptr()))
    del arg271_1
    del arg272_1
    del arg737_1
    del arg738_1
    del buf257
    # Source Nodes: [out_206], Original ATen: [aten.convolution]
    buf259 = extern_kernels.convolution(buf258, arg273_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf259, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg273_1
    buf260 = buf259; del buf259  # reuse
    buf261 = buf247; del buf247  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_91(c_void_p(buf260.data_ptr()), c_void_p(arg740_1.data_ptr()), c_void_p(arg741_1.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(buf261.data_ptr()))
    del arg274_1
    del arg275_1
    del arg276_1
    del arg740_1
    del arg741_1
    # Source Nodes: [out_207, out_208, x_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf262 = extern_kernels.convolution(buf260, buf261, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf262, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf263 = buf262; del buf262  # reuse
    buf264 = reinterpret_tensor(buf251, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf251  # reuse
    buf265 = reinterpret_tensor(buf264, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf264  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_92(c_void_p(buf263.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(arg743_1.data_ptr()), c_void_p(arg744_1.data_ptr()), c_void_p(arg277_1.data_ptr()), c_void_p(arg278_1.data_ptr()))
    del arg277_1
    del arg278_1
    del arg743_1
    del arg744_1
    # Source Nodes: [x_gap_85, x_gap_86, x_gap_87], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf266 = extern_kernels.convolution(buf265, arg279_1, arg280_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf266, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg279_1
    del arg280_1
    buf267 = buf266; del buf266  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_93(c_void_p(buf267.data_ptr()), c_void_p(arg746_1.data_ptr()), c_void_p(arg747_1.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(arg282_1.data_ptr()))
    del arg281_1
    del arg282_1
    del arg746_1
    del arg747_1
    # Source Nodes: [x_attn_34, x_gap_88, x_gap_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf268 = extern_kernels.convolution(buf267, arg283_1, arg284_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf268, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg283_1
    del arg284_1
    del buf267
    buf269 = buf255; del buf255  # reuse
    buf270 = buf260; del buf260  # reuse
    cpp_fused__softmax_mul_sum_94(c_void_p(buf268.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()))
    del buf263
    del buf268
    # Source Nodes: [mul_17, out_209, out_214], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf271 = extern_kernels.convolution(buf270, arg285_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf271, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg285_1
    del buf270
    buf272 = buf258; del buf258  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_95(c_void_p(buf272.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(arg749_1.data_ptr()), c_void_p(arg750_1.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg287_1.data_ptr()))
    del arg286_1
    del arg287_1
    del arg749_1
    del arg750_1
    del buf271
    # Source Nodes: [out_218], Original ATen: [aten.convolution]
    buf273 = extern_kernels.convolution(buf272, arg288_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf273, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg288_1
    buf274 = buf273; del buf273  # reuse
    buf275 = buf261; del buf261  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_96(c_void_p(buf274.data_ptr()), c_void_p(arg752_1.data_ptr()), c_void_p(arg753_1.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(buf275.data_ptr()))
    del arg289_1
    del arg290_1
    del arg291_1
    del arg752_1
    del arg753_1
    # Source Nodes: [out_219, out_220, x_150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf276 = extern_kernels.convolution(buf274, buf275, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf276, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf277 = buf276; del buf276  # reuse
    buf278 = reinterpret_tensor(buf265, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf265  # reuse
    buf279 = reinterpret_tensor(buf278, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf278  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_97(c_void_p(buf277.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(arg755_1.data_ptr()), c_void_p(arg756_1.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(arg293_1.data_ptr()))
    del arg292_1
    del arg293_1
    del arg755_1
    del arg756_1
    # Source Nodes: [x_gap_90, x_gap_91, x_gap_92], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf280 = extern_kernels.convolution(buf279, arg294_1, arg295_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf280, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg294_1
    del arg295_1
    buf281 = buf280; del buf280  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_98(c_void_p(buf281.data_ptr()), c_void_p(arg758_1.data_ptr()), c_void_p(arg759_1.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(arg297_1.data_ptr()))
    del arg296_1
    del arg297_1
    del arg758_1
    del arg759_1
    # Source Nodes: [x_attn_36, x_gap_93, x_gap_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf282 = extern_kernels.convolution(buf281, arg298_1, arg299_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf282, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg298_1
    del arg299_1
    del buf281
    buf283 = buf269; del buf269  # reuse
    buf284 = buf274; del buf274  # reuse
    cpp_fused__softmax_mul_sum_99(c_void_p(buf282.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()))
    del buf277
    del buf282
    # Source Nodes: [mul_18, out_221, out_226], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf285 = extern_kernels.convolution(buf284, arg300_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf285, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg300_1
    del buf284
    buf286 = buf272; del buf272  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_100(c_void_p(buf286.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(arg761_1.data_ptr()), c_void_p(arg762_1.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(arg302_1.data_ptr()))
    del arg301_1
    del arg302_1
    del arg761_1
    del arg762_1
    del buf285
    # Source Nodes: [out_230], Original ATen: [aten.convolution]
    buf287 = extern_kernels.convolution(buf286, arg303_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf287, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg303_1
    buf288 = buf287; del buf287  # reuse
    buf289 = buf275; del buf275  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_101(c_void_p(buf288.data_ptr()), c_void_p(arg764_1.data_ptr()), c_void_p(arg765_1.data_ptr()), c_void_p(arg304_1.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(buf289.data_ptr()))
    del arg304_1
    del arg305_1
    del arg306_1
    del arg764_1
    del arg765_1
    # Source Nodes: [out_231, out_232, x_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf290 = extern_kernels.convolution(buf288, buf289, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf290, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf291 = buf290; del buf290  # reuse
    buf292 = reinterpret_tensor(buf279, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf279  # reuse
    buf293 = reinterpret_tensor(buf292, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf292  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_102(c_void_p(buf291.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(arg767_1.data_ptr()), c_void_p(arg768_1.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg308_1.data_ptr()))
    del arg307_1
    del arg308_1
    del arg767_1
    del arg768_1
    # Source Nodes: [x_gap_95, x_gap_96, x_gap_97], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf294 = extern_kernels.convolution(buf293, arg309_1, arg310_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf294, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg309_1
    del arg310_1
    buf295 = buf294; del buf294  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_103(c_void_p(buf295.data_ptr()), c_void_p(arg770_1.data_ptr()), c_void_p(arg771_1.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(arg312_1.data_ptr()))
    del arg311_1
    del arg312_1
    del arg770_1
    del arg771_1
    # Source Nodes: [x_attn_38, x_gap_98, x_gap_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf296 = extern_kernels.convolution(buf295, arg313_1, arg314_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf296, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg313_1
    del arg314_1
    del buf295
    buf297 = buf283; del buf283  # reuse
    buf298 = buf288; del buf288  # reuse
    cpp_fused__softmax_mul_sum_104(c_void_p(buf296.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()))
    del buf291
    del buf296
    # Source Nodes: [mul_19, out_233, out_238], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf299 = extern_kernels.convolution(buf298, arg315_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf299, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg315_1
    del buf298
    buf300 = buf286; del buf286  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_105(c_void_p(buf300.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(arg773_1.data_ptr()), c_void_p(arg774_1.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(arg317_1.data_ptr()))
    del arg316_1
    del arg317_1
    del arg773_1
    del arg774_1
    del buf299
    # Source Nodes: [out_242], Original ATen: [aten.convolution]
    buf301 = extern_kernels.convolution(buf300, arg318_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf301, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg318_1
    buf302 = buf301; del buf301  # reuse
    buf303 = buf289; del buf289  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_106(c_void_p(buf302.data_ptr()), c_void_p(arg776_1.data_ptr()), c_void_p(arg777_1.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(arg321_1.data_ptr()), c_void_p(buf303.data_ptr()))
    del arg319_1
    del arg320_1
    del arg321_1
    del arg776_1
    del arg777_1
    # Source Nodes: [out_243, out_244, x_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf304 = extern_kernels.convolution(buf302, buf303, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf304, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf305 = buf304; del buf304  # reuse
    buf306 = reinterpret_tensor(buf293, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf293  # reuse
    buf307 = reinterpret_tensor(buf306, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf306  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_107(c_void_p(buf305.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(arg779_1.data_ptr()), c_void_p(arg780_1.data_ptr()), c_void_p(arg322_1.data_ptr()), c_void_p(arg323_1.data_ptr()))
    del arg322_1
    del arg323_1
    del arg779_1
    del arg780_1
    # Source Nodes: [x_gap_100, x_gap_101, x_gap_102], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf308 = extern_kernels.convolution(buf307, arg324_1, arg325_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf308, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg324_1
    del arg325_1
    buf309 = buf308; del buf308  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_108(c_void_p(buf309.data_ptr()), c_void_p(arg782_1.data_ptr()), c_void_p(arg783_1.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(arg327_1.data_ptr()))
    del arg326_1
    del arg327_1
    del arg782_1
    del arg783_1
    # Source Nodes: [x_attn_40, x_gap_103, x_gap_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf310 = extern_kernels.convolution(buf309, arg328_1, arg329_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf310, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg328_1
    del arg329_1
    del buf309
    buf311 = buf297; del buf297  # reuse
    buf312 = buf302; del buf302  # reuse
    cpp_fused__softmax_mul_sum_109(c_void_p(buf310.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()))
    del buf305
    del buf310
    # Source Nodes: [mul_20, out_245, out_250], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf313 = extern_kernels.convolution(buf312, arg330_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf313, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg330_1
    del buf312
    buf314 = buf300; del buf300  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_110(c_void_p(buf314.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(arg785_1.data_ptr()), c_void_p(arg786_1.data_ptr()), c_void_p(arg331_1.data_ptr()), c_void_p(arg332_1.data_ptr()))
    del arg331_1
    del arg332_1
    del arg785_1
    del arg786_1
    del buf313
    # Source Nodes: [out_254], Original ATen: [aten.convolution]
    buf315 = extern_kernels.convolution(buf314, arg333_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf315, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg333_1
    buf316 = buf315; del buf315  # reuse
    buf317 = buf303; del buf303  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_111(c_void_p(buf316.data_ptr()), c_void_p(arg788_1.data_ptr()), c_void_p(arg789_1.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(buf317.data_ptr()))
    del arg334_1
    del arg335_1
    del arg336_1
    del arg788_1
    del arg789_1
    # Source Nodes: [out_255, out_256, x_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf318 = extern_kernels.convolution(buf316, buf317, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf318, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf319 = buf318; del buf318  # reuse
    buf320 = reinterpret_tensor(buf307, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf307  # reuse
    buf321 = reinterpret_tensor(buf320, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf320  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_112(c_void_p(buf319.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(arg791_1.data_ptr()), c_void_p(arg792_1.data_ptr()), c_void_p(arg337_1.data_ptr()), c_void_p(arg338_1.data_ptr()))
    del arg337_1
    del arg338_1
    del arg791_1
    del arg792_1
    # Source Nodes: [x_gap_105, x_gap_106, x_gap_107], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf322 = extern_kernels.convolution(buf321, arg339_1, arg340_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf322, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg339_1
    del arg340_1
    buf323 = buf322; del buf322  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_113(c_void_p(buf323.data_ptr()), c_void_p(arg794_1.data_ptr()), c_void_p(arg795_1.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(arg342_1.data_ptr()))
    del arg341_1
    del arg342_1
    del arg794_1
    del arg795_1
    # Source Nodes: [x_attn_42, x_gap_108, x_gap_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf324 = extern_kernels.convolution(buf323, arg343_1, arg344_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf324, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg343_1
    del arg344_1
    del buf323
    buf325 = buf311; del buf311  # reuse
    buf326 = buf316; del buf316  # reuse
    cpp_fused__softmax_mul_sum_114(c_void_p(buf324.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()))
    del buf319
    del buf324
    # Source Nodes: [mul_21, out_257, out_262], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf327 = extern_kernels.convolution(buf326, arg345_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf327, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg345_1
    del buf326
    buf328 = buf314; del buf314  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_115(c_void_p(buf328.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(arg797_1.data_ptr()), c_void_p(arg798_1.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(arg347_1.data_ptr()))
    del arg346_1
    del arg347_1
    del arg797_1
    del arg798_1
    del buf327
    # Source Nodes: [out_266], Original ATen: [aten.convolution]
    buf329 = extern_kernels.convolution(buf328, arg348_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf329, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg348_1
    buf330 = buf329; del buf329  # reuse
    buf331 = buf317; del buf317  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_116(c_void_p(buf330.data_ptr()), c_void_p(arg800_1.data_ptr()), c_void_p(arg801_1.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(arg350_1.data_ptr()), c_void_p(arg351_1.data_ptr()), c_void_p(buf331.data_ptr()))
    del arg349_1
    del arg350_1
    del arg351_1
    del arg800_1
    del arg801_1
    # Source Nodes: [out_267, out_268, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf332 = extern_kernels.convolution(buf330, buf331, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf332, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf333 = buf332; del buf332  # reuse
    buf334 = reinterpret_tensor(buf321, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf321  # reuse
    buf335 = reinterpret_tensor(buf334, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf334  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_117(c_void_p(buf333.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(arg803_1.data_ptr()), c_void_p(arg804_1.data_ptr()), c_void_p(arg352_1.data_ptr()), c_void_p(arg353_1.data_ptr()))
    del arg352_1
    del arg353_1
    del arg803_1
    del arg804_1
    # Source Nodes: [x_gap_110, x_gap_111, x_gap_112], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf336 = extern_kernels.convolution(buf335, arg354_1, arg355_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf336, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg354_1
    del arg355_1
    buf337 = buf336; del buf336  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_118(c_void_p(buf337.data_ptr()), c_void_p(arg806_1.data_ptr()), c_void_p(arg807_1.data_ptr()), c_void_p(arg356_1.data_ptr()), c_void_p(arg357_1.data_ptr()))
    del arg356_1
    del arg357_1
    del arg806_1
    del arg807_1
    # Source Nodes: [x_attn_44, x_gap_113, x_gap_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf338 = extern_kernels.convolution(buf337, arg358_1, arg359_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf338, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg358_1
    del arg359_1
    del buf337
    buf339 = buf325; del buf325  # reuse
    buf340 = buf330; del buf330  # reuse
    cpp_fused__softmax_mul_sum_119(c_void_p(buf338.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()))
    del buf333
    del buf338
    # Source Nodes: [mul_22, out_269, out_274], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf341 = extern_kernels.convolution(buf340, arg360_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf341, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg360_1
    del buf340
    buf342 = buf328; del buf328  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_120(c_void_p(buf342.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(arg809_1.data_ptr()), c_void_p(arg810_1.data_ptr()), c_void_p(arg361_1.data_ptr()), c_void_p(arg362_1.data_ptr()))
    del arg361_1
    del arg362_1
    del arg809_1
    del arg810_1
    del buf341
    # Source Nodes: [out_278], Original ATen: [aten.convolution]
    buf343 = extern_kernels.convolution(buf342, arg363_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf343, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg363_1
    buf344 = buf343; del buf343  # reuse
    buf345 = buf331; del buf331  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_121(c_void_p(buf344.data_ptr()), c_void_p(arg812_1.data_ptr()), c_void_p(arg813_1.data_ptr()), c_void_p(arg364_1.data_ptr()), c_void_p(arg365_1.data_ptr()), c_void_p(arg366_1.data_ptr()), c_void_p(buf345.data_ptr()))
    del arg364_1
    del arg365_1
    del arg366_1
    del arg812_1
    del arg813_1
    # Source Nodes: [out_279, out_280, x_190], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf346 = extern_kernels.convolution(buf344, buf345, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf346, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf347 = buf346; del buf346  # reuse
    buf348 = reinterpret_tensor(buf335, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf335  # reuse
    buf349 = reinterpret_tensor(buf348, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf348  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_122(c_void_p(buf347.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(arg815_1.data_ptr()), c_void_p(arg816_1.data_ptr()), c_void_p(arg367_1.data_ptr()), c_void_p(arg368_1.data_ptr()))
    del arg367_1
    del arg368_1
    del arg815_1
    del arg816_1
    # Source Nodes: [x_gap_115, x_gap_116, x_gap_117], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf350 = extern_kernels.convolution(buf349, arg369_1, arg370_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf350, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg369_1
    del arg370_1
    buf351 = buf350; del buf350  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_123(c_void_p(buf351.data_ptr()), c_void_p(arg818_1.data_ptr()), c_void_p(arg819_1.data_ptr()), c_void_p(arg371_1.data_ptr()), c_void_p(arg372_1.data_ptr()))
    del arg371_1
    del arg372_1
    del arg818_1
    del arg819_1
    # Source Nodes: [x_attn_46, x_gap_118, x_gap_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf352 = extern_kernels.convolution(buf351, arg373_1, arg374_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf352, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg373_1
    del arg374_1
    del buf351
    buf353 = buf339; del buf339  # reuse
    buf354 = buf344; del buf344  # reuse
    cpp_fused__softmax_mul_sum_124(c_void_p(buf352.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()))
    del buf347
    del buf352
    # Source Nodes: [mul_23, out_281, out_286], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf355 = extern_kernels.convolution(buf354, arg375_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf355, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg375_1
    del buf354
    buf356 = buf342; del buf342  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_125(c_void_p(buf356.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(arg821_1.data_ptr()), c_void_p(arg822_1.data_ptr()), c_void_p(arg376_1.data_ptr()), c_void_p(arg377_1.data_ptr()))
    del arg376_1
    del arg377_1
    del arg821_1
    del arg822_1
    del buf355
    # Source Nodes: [out_290], Original ATen: [aten.convolution]
    buf357 = extern_kernels.convolution(buf356, arg378_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf357, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg378_1
    buf358 = buf357; del buf357  # reuse
    buf359 = buf345; del buf345  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_126(c_void_p(buf358.data_ptr()), c_void_p(arg824_1.data_ptr()), c_void_p(arg825_1.data_ptr()), c_void_p(arg379_1.data_ptr()), c_void_p(arg380_1.data_ptr()), c_void_p(arg381_1.data_ptr()), c_void_p(buf359.data_ptr()))
    del arg379_1
    del arg380_1
    del arg381_1
    del arg824_1
    del arg825_1
    # Source Nodes: [out_291, out_292, x_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf360 = extern_kernels.convolution(buf358, buf359, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf360, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf361 = buf360; del buf360  # reuse
    buf362 = reinterpret_tensor(buf349, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf349  # reuse
    buf363 = reinterpret_tensor(buf362, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf362  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_127(c_void_p(buf361.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(arg827_1.data_ptr()), c_void_p(arg828_1.data_ptr()), c_void_p(arg382_1.data_ptr()), c_void_p(arg383_1.data_ptr()))
    del arg382_1
    del arg383_1
    del arg827_1
    del arg828_1
    # Source Nodes: [x_gap_120, x_gap_121, x_gap_122], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf364 = extern_kernels.convolution(buf363, arg384_1, arg385_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf364, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg384_1
    del arg385_1
    buf365 = buf364; del buf364  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_128(c_void_p(buf365.data_ptr()), c_void_p(arg830_1.data_ptr()), c_void_p(arg831_1.data_ptr()), c_void_p(arg386_1.data_ptr()), c_void_p(arg387_1.data_ptr()))
    del arg386_1
    del arg387_1
    del arg830_1
    del arg831_1
    # Source Nodes: [x_attn_48, x_gap_123, x_gap_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf366 = extern_kernels.convolution(buf365, arg388_1, arg389_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf366, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg388_1
    del arg389_1
    del buf365
    buf367 = buf353; del buf353  # reuse
    buf368 = buf358; del buf358  # reuse
    cpp_fused__softmax_mul_sum_129(c_void_p(buf366.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()))
    del buf361
    del buf366
    # Source Nodes: [mul_24, out_293, out_298], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf369 = extern_kernels.convolution(buf368, arg390_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf369, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg390_1
    del buf368
    buf370 = buf356; del buf356  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_130(c_void_p(buf370.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(arg833_1.data_ptr()), c_void_p(arg834_1.data_ptr()), c_void_p(arg391_1.data_ptr()), c_void_p(arg392_1.data_ptr()))
    del arg391_1
    del arg392_1
    del arg833_1
    del arg834_1
    del buf369
    # Source Nodes: [out_302], Original ATen: [aten.convolution]
    buf371 = extern_kernels.convolution(buf370, arg393_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf371, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg393_1
    buf372 = buf371; del buf371  # reuse
    buf373 = buf359; del buf359  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_131(c_void_p(buf372.data_ptr()), c_void_p(arg836_1.data_ptr()), c_void_p(arg837_1.data_ptr()), c_void_p(arg394_1.data_ptr()), c_void_p(arg395_1.data_ptr()), c_void_p(arg396_1.data_ptr()), c_void_p(buf373.data_ptr()))
    del arg394_1
    del arg395_1
    del arg396_1
    del arg836_1
    del arg837_1
    # Source Nodes: [out_303, out_304, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf374 = extern_kernels.convolution(buf372, buf373, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf374, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf375 = buf374; del buf374  # reuse
    buf376 = reinterpret_tensor(buf363, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf363  # reuse
    buf377 = reinterpret_tensor(buf376, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf376  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_132(c_void_p(buf375.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(arg839_1.data_ptr()), c_void_p(arg840_1.data_ptr()), c_void_p(arg397_1.data_ptr()), c_void_p(arg398_1.data_ptr()))
    del arg397_1
    del arg398_1
    del arg839_1
    del arg840_1
    # Source Nodes: [x_gap_125, x_gap_126, x_gap_127], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf378 = extern_kernels.convolution(buf377, arg399_1, arg400_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf378, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg399_1
    del arg400_1
    buf379 = buf378; del buf378  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_133(c_void_p(buf379.data_ptr()), c_void_p(arg842_1.data_ptr()), c_void_p(arg843_1.data_ptr()), c_void_p(arg401_1.data_ptr()), c_void_p(arg402_1.data_ptr()))
    del arg401_1
    del arg402_1
    del arg842_1
    del arg843_1
    # Source Nodes: [x_attn_50, x_gap_128, x_gap_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf380 = extern_kernels.convolution(buf379, arg403_1, arg404_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf380, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg403_1
    del arg404_1
    del buf379
    buf381 = buf367; del buf367  # reuse
    buf382 = buf372; del buf372  # reuse
    cpp_fused__softmax_mul_sum_134(c_void_p(buf380.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()))
    del buf375
    del buf380
    # Source Nodes: [mul_25, out_305, out_310], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf383 = extern_kernels.convolution(buf382, arg405_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf383, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg405_1
    del buf382
    buf384 = buf370; del buf370  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_135(c_void_p(buf384.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(arg845_1.data_ptr()), c_void_p(arg846_1.data_ptr()), c_void_p(arg406_1.data_ptr()), c_void_p(arg407_1.data_ptr()))
    del arg406_1
    del arg407_1
    del arg845_1
    del arg846_1
    del buf383
    # Source Nodes: [out_314], Original ATen: [aten.convolution]
    buf385 = extern_kernels.convolution(buf384, arg408_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf385, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg408_1
    buf386 = buf385; del buf385  # reuse
    buf387 = buf373; del buf373  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_136(c_void_p(buf386.data_ptr()), c_void_p(arg848_1.data_ptr()), c_void_p(arg849_1.data_ptr()), c_void_p(arg409_1.data_ptr()), c_void_p(arg410_1.data_ptr()), c_void_p(arg411_1.data_ptr()), c_void_p(buf387.data_ptr()))
    del arg409_1
    del arg410_1
    del arg411_1
    del arg848_1
    del arg849_1
    # Source Nodes: [out_315, out_316, x_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf388 = extern_kernels.convolution(buf386, buf387, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf388, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf389 = buf388; del buf388  # reuse
    buf390 = reinterpret_tensor(buf377, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf377  # reuse
    buf391 = reinterpret_tensor(buf390, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf390  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_137(c_void_p(buf389.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(arg851_1.data_ptr()), c_void_p(arg852_1.data_ptr()), c_void_p(arg412_1.data_ptr()), c_void_p(arg413_1.data_ptr()))
    del arg412_1
    del arg413_1
    del arg851_1
    del arg852_1
    # Source Nodes: [x_gap_130, x_gap_131, x_gap_132], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf392 = extern_kernels.convolution(buf391, arg414_1, arg415_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf392, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg414_1
    del arg415_1
    buf393 = buf392; del buf392  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_138(c_void_p(buf393.data_ptr()), c_void_p(arg854_1.data_ptr()), c_void_p(arg855_1.data_ptr()), c_void_p(arg416_1.data_ptr()), c_void_p(arg417_1.data_ptr()))
    del arg416_1
    del arg417_1
    del arg854_1
    del arg855_1
    # Source Nodes: [x_attn_52, x_gap_133, x_gap_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf394 = extern_kernels.convolution(buf393, arg418_1, arg419_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf394, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg418_1
    del arg419_1
    del buf393
    buf395 = buf381; del buf381  # reuse
    buf396 = buf386; del buf386  # reuse
    cpp_fused__softmax_mul_sum_139(c_void_p(buf394.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()))
    del buf389
    del buf394
    # Source Nodes: [mul_26, out_317, out_322], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf397 = extern_kernels.convolution(buf396, arg420_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf397, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg420_1
    del buf396
    buf398 = buf384; del buf384  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_140(c_void_p(buf398.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(arg857_1.data_ptr()), c_void_p(arg858_1.data_ptr()), c_void_p(arg421_1.data_ptr()), c_void_p(arg422_1.data_ptr()))
    del arg421_1
    del arg422_1
    del arg857_1
    del arg858_1
    del buf397
    # Source Nodes: [out_326], Original ATen: [aten.convolution]
    buf399 = extern_kernels.convolution(buf398, arg423_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf399, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg423_1
    buf400 = buf399; del buf399  # reuse
    buf401 = buf387; del buf387  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_141(c_void_p(buf400.data_ptr()), c_void_p(arg860_1.data_ptr()), c_void_p(arg861_1.data_ptr()), c_void_p(arg424_1.data_ptr()), c_void_p(arg425_1.data_ptr()), c_void_p(arg426_1.data_ptr()), c_void_p(buf401.data_ptr()))
    del arg424_1
    del arg425_1
    del arg426_1
    del arg860_1
    del arg861_1
    # Source Nodes: [out_327, out_328, x_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf402 = extern_kernels.convolution(buf400, buf401, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf402, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf403 = buf402; del buf402  # reuse
    buf404 = reinterpret_tensor(buf391, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf391  # reuse
    buf405 = reinterpret_tensor(buf404, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf404  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_142(c_void_p(buf403.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(arg863_1.data_ptr()), c_void_p(arg864_1.data_ptr()), c_void_p(arg427_1.data_ptr()), c_void_p(arg428_1.data_ptr()))
    del arg427_1
    del arg428_1
    del arg863_1
    del arg864_1
    # Source Nodes: [x_gap_135, x_gap_136, x_gap_137], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf406 = extern_kernels.convolution(buf405, arg429_1, arg430_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf406, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg429_1
    del arg430_1
    buf407 = buf406; del buf406  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_143(c_void_p(buf407.data_ptr()), c_void_p(arg866_1.data_ptr()), c_void_p(arg867_1.data_ptr()), c_void_p(arg431_1.data_ptr()), c_void_p(arg432_1.data_ptr()))
    del arg431_1
    del arg432_1
    del arg866_1
    del arg867_1
    # Source Nodes: [x_attn_54, x_gap_138, x_gap_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf408 = extern_kernels.convolution(buf407, arg433_1, arg434_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf408, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg433_1
    del arg434_1
    del buf407
    buf409 = buf395; del buf395  # reuse
    buf410 = buf400; del buf400  # reuse
    cpp_fused__softmax_mul_sum_144(c_void_p(buf408.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf410.data_ptr()))
    del buf403
    del buf408
    # Source Nodes: [mul_27, out_329, out_334], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf411 = extern_kernels.convolution(buf410, arg435_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf411, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg435_1
    del buf410
    buf412 = buf398; del buf398  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_145(c_void_p(buf412.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(arg869_1.data_ptr()), c_void_p(arg870_1.data_ptr()), c_void_p(arg436_1.data_ptr()), c_void_p(arg437_1.data_ptr()))
    del arg436_1
    del arg437_1
    del arg869_1
    del arg870_1
    del buf411
    # Source Nodes: [out_338], Original ATen: [aten.convolution]
    buf413 = extern_kernels.convolution(buf412, arg438_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf413, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg438_1
    buf414 = buf413; del buf413  # reuse
    buf415 = buf401; del buf401  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_146(c_void_p(buf414.data_ptr()), c_void_p(arg872_1.data_ptr()), c_void_p(arg873_1.data_ptr()), c_void_p(arg439_1.data_ptr()), c_void_p(arg440_1.data_ptr()), c_void_p(arg441_1.data_ptr()), c_void_p(buf415.data_ptr()))
    del arg439_1
    del arg440_1
    del arg441_1
    del arg872_1
    del arg873_1
    # Source Nodes: [out_339, out_340, x_230], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf416 = extern_kernels.convolution(buf414, buf415, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf416, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf417 = buf416; del buf416  # reuse
    buf418 = reinterpret_tensor(buf405, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf405  # reuse
    buf419 = reinterpret_tensor(buf418, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf418  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_147(c_void_p(buf417.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(arg875_1.data_ptr()), c_void_p(arg876_1.data_ptr()), c_void_p(arg442_1.data_ptr()), c_void_p(arg443_1.data_ptr()))
    del arg442_1
    del arg443_1
    del arg875_1
    del arg876_1
    # Source Nodes: [x_gap_140, x_gap_141, x_gap_142], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf420 = extern_kernels.convolution(buf419, arg444_1, arg445_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf420, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg444_1
    del arg445_1
    buf421 = buf420; del buf420  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_148(c_void_p(buf421.data_ptr()), c_void_p(arg878_1.data_ptr()), c_void_p(arg879_1.data_ptr()), c_void_p(arg446_1.data_ptr()), c_void_p(arg447_1.data_ptr()))
    del arg446_1
    del arg447_1
    del arg878_1
    del arg879_1
    # Source Nodes: [x_attn_56, x_gap_143, x_gap_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf422 = extern_kernels.convolution(buf421, arg448_1, arg449_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf422, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg448_1
    del arg449_1
    del buf421
    buf423 = buf409; del buf409  # reuse
    buf424 = buf414; del buf414  # reuse
    cpp_fused__softmax_mul_sum_149(c_void_p(buf422.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()))
    del buf417
    del buf422
    # Source Nodes: [mul_28, out_341, out_346], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf425 = extern_kernels.convolution(buf424, arg450_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf425, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg450_1
    del buf424
    buf426 = buf412; del buf412  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_150(c_void_p(buf426.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(arg881_1.data_ptr()), c_void_p(arg882_1.data_ptr()), c_void_p(arg451_1.data_ptr()), c_void_p(arg452_1.data_ptr()))
    del arg451_1
    del arg452_1
    del arg881_1
    del arg882_1
    del buf425
    # Source Nodes: [out_350], Original ATen: [aten.convolution]
    buf427 = extern_kernels.convolution(buf426, arg453_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf427, (8, 256, 16, 16), (65536, 1, 4096, 256))
    del arg453_1
    buf428 = buf427; del buf427  # reuse
    buf429 = buf415; del buf415  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_151(c_void_p(buf428.data_ptr()), c_void_p(arg884_1.data_ptr()), c_void_p(arg885_1.data_ptr()), c_void_p(arg454_1.data_ptr()), c_void_p(arg455_1.data_ptr()), c_void_p(arg456_1.data_ptr()), c_void_p(buf429.data_ptr()))
    del arg454_1
    del arg455_1
    del arg456_1
    del arg884_1
    del arg885_1
    # Source Nodes: [out_351, out_352, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf430 = extern_kernels.convolution(buf428, buf429, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf430, (8, 512, 16, 16), (131072, 1, 8192, 512))
    del buf429
    buf431 = buf430; del buf430  # reuse
    buf432 = reinterpret_tensor(buf419, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf419  # reuse
    buf433 = reinterpret_tensor(buf432, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf432  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_152(c_void_p(buf431.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(arg887_1.data_ptr()), c_void_p(arg888_1.data_ptr()), c_void_p(arg457_1.data_ptr()), c_void_p(arg458_1.data_ptr()))
    del arg457_1
    del arg458_1
    del arg887_1
    del arg888_1
    # Source Nodes: [x_gap_145, x_gap_146, x_gap_147], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf434 = extern_kernels.convolution(buf433, arg459_1, arg460_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf434, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg459_1
    del arg460_1
    del buf433
    buf435 = buf434; del buf434  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_153(c_void_p(buf435.data_ptr()), c_void_p(arg890_1.data_ptr()), c_void_p(arg891_1.data_ptr()), c_void_p(arg461_1.data_ptr()), c_void_p(arg462_1.data_ptr()))
    del arg461_1
    del arg462_1
    del arg890_1
    del arg891_1
    # Source Nodes: [x_attn_58, x_gap_148, x_gap_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf436 = extern_kernels.convolution(buf435, arg463_1, arg464_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf436, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg463_1
    del arg464_1
    del buf435
    buf437 = buf423; del buf423  # reuse
    buf438 = buf428; del buf428  # reuse
    cpp_fused__softmax_mul_sum_154(c_void_p(buf436.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()))
    del buf431
    del buf436
    # Source Nodes: [mul_29, out_353, out_358], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf439 = extern_kernels.convolution(buf438, arg465_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf439, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    del arg465_1
    buf440 = buf426; del buf426  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_155(c_void_p(buf440.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(arg893_1.data_ptr()), c_void_p(arg894_1.data_ptr()), c_void_p(arg466_1.data_ptr()), c_void_p(arg467_1.data_ptr()))
    del arg466_1
    del arg467_1
    del arg893_1
    del arg894_1
    del buf439
    # Source Nodes: [out_362], Original ATen: [aten.convolution]
    buf441 = extern_kernels.convolution(buf440, arg468_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf441, (8, 512, 16, 16), (131072, 1, 8192, 512))
    del arg468_1
    buf442 = buf441; del buf441  # reuse
    buf443 = empty_strided((1024, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_156(c_void_p(buf442.data_ptr()), c_void_p(arg896_1.data_ptr()), c_void_p(arg897_1.data_ptr()), c_void_p(arg469_1.data_ptr()), c_void_p(arg470_1.data_ptr()), c_void_p(arg471_1.data_ptr()), c_void_p(buf443.data_ptr()))
    del arg469_1
    del arg470_1
    del arg471_1
    del arg896_1
    del arg897_1
    # Source Nodes: [out_363, out_364, x_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf444 = extern_kernels.convolution(buf442, buf443, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf444, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    buf445 = buf444; del buf444  # reuse
    buf446 = reinterpret_tensor(buf437, (8, 512, 1, 1), (512, 1, 4096, 4096), 0); del buf437  # reuse
    buf447 = reinterpret_tensor(buf446, (8, 512, 1, 1), (512, 1, 512, 512), 0); del buf446  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_157(c_void_p(buf445.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(arg899_1.data_ptr()), c_void_p(arg900_1.data_ptr()), c_void_p(arg472_1.data_ptr()), c_void_p(arg473_1.data_ptr()))
    del arg472_1
    del arg473_1
    del arg899_1
    del arg900_1
    # Source Nodes: [x_gap_150, x_gap_151, x_gap_152], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf448 = extern_kernels.convolution(buf447, arg474_1, arg475_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf448, (8, 256, 1, 1), (256, 1, 256, 256))
    del arg474_1
    del arg475_1
    buf449 = buf448; del buf448  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_158(c_void_p(buf449.data_ptr()), c_void_p(arg902_1.data_ptr()), c_void_p(arg903_1.data_ptr()), c_void_p(arg476_1.data_ptr()), c_void_p(arg477_1.data_ptr()))
    del arg476_1
    del arg477_1
    del arg902_1
    del arg903_1
    # Source Nodes: [x_attn_60, x_gap_153, x_gap_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf450 = extern_kernels.convolution(buf449, arg478_1, arg479_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf450, (8, 1024, 1, 1), (1024, 1, 1024, 1024))
    del arg478_1
    del arg479_1
    del buf449
    buf451 = empty_strided((8, 2, 1, 512), (1024, 512, 8192, 1), device='cpu', dtype=torch.float32)
    buf452 = buf442; del buf442  # reuse
    buf453 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_avg_pool2d_mul_sum_159(c_void_p(buf450.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()))
    del buf445
    del buf450
    del buf452
    # Source Nodes: [out_371], Original ATen: [aten.convolution]
    buf454 = extern_kernels.convolution(buf453, arg480_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf454, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    del arg480_1
    del buf453
    buf455 = reinterpret_tensor(buf438, (8, 1024, 8, 8), (65536, 1, 8192, 1024), 0); del buf438  # reuse
    cpp_fused_avg_pool2d_160(c_void_p(buf440.data_ptr()), c_void_p(buf455.data_ptr()))
    del buf440
    # Source Nodes: [getattr_l__mod___layer4___0___downsample_0, getattr_l__mod___layer4___0___downsample_1], Original ATen: [aten.avg_pool2d, aten.convolution]
    buf456 = extern_kernels.convolution(buf455, arg483_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf456, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    del arg483_1
    del buf455
    buf457 = buf454; del buf454  # reuse
    buf458 = buf457; del buf457  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_161(c_void_p(buf458.data_ptr()), c_void_p(arg905_1.data_ptr()), c_void_p(arg906_1.data_ptr()), c_void_p(arg481_1.data_ptr()), c_void_p(arg482_1.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(arg908_1.data_ptr()), c_void_p(arg909_1.data_ptr()), c_void_p(arg484_1.data_ptr()), c_void_p(arg485_1.data_ptr()))
    del arg481_1
    del arg482_1
    del arg484_1
    del arg485_1
    del arg905_1
    del arg906_1
    del arg908_1
    del arg909_1
    del buf456
    # Source Nodes: [out_375, shortcut_35], Original ATen: [aten.convolution, aten.relu]
    buf459 = extern_kernels.convolution(buf458, arg486_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf459, (8, 512, 8, 8), (32768, 1, 4096, 512))
    del arg486_1
    buf460 = buf459; del buf459  # reuse
    buf461 = buf443; del buf443  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_162(c_void_p(buf460.data_ptr()), c_void_p(arg911_1.data_ptr()), c_void_p(arg912_1.data_ptr()), c_void_p(arg487_1.data_ptr()), c_void_p(arg488_1.data_ptr()), c_void_p(arg489_1.data_ptr()), c_void_p(buf461.data_ptr()))
    del arg487_1
    del arg488_1
    del arg489_1
    del arg911_1
    del arg912_1
    # Source Nodes: [out_376, out_377, x_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf462 = extern_kernels.convolution(buf460, buf461, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf462, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
    buf463 = buf462; del buf462  # reuse
    buf464 = reinterpret_tensor(buf447, (8, 512, 1, 1), (512, 1, 4096, 4096), 0); del buf447  # reuse
    buf465 = reinterpret_tensor(buf464, (8, 512, 1, 1), (512, 1, 512, 512), 0); del buf464  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_163(c_void_p(buf463.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(arg914_1.data_ptr()), c_void_p(arg915_1.data_ptr()), c_void_p(arg490_1.data_ptr()), c_void_p(arg491_1.data_ptr()))
    del arg490_1
    del arg491_1
    del arg914_1
    del arg915_1
    # Source Nodes: [x_gap_155, x_gap_156, x_gap_157], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf466 = extern_kernels.convolution(buf465, arg492_1, arg493_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf466, (8, 256, 1, 1), (256, 1, 256, 256))
    del arg492_1
    del arg493_1
    buf467 = buf466; del buf466  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_164(c_void_p(buf467.data_ptr()), c_void_p(arg917_1.data_ptr()), c_void_p(arg918_1.data_ptr()), c_void_p(arg494_1.data_ptr()), c_void_p(arg495_1.data_ptr()))
    del arg494_1
    del arg495_1
    del arg917_1
    del arg918_1
    # Source Nodes: [x_attn_62, x_gap_158, x_gap_159], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf468 = extern_kernels.convolution(buf467, arg496_1, arg497_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf468, (8, 1024, 1, 1), (1024, 1, 1024, 1024))
    del arg496_1
    del arg497_1
    del buf467
    buf469 = buf451; del buf451  # reuse
    buf470 = buf460; del buf460  # reuse
    cpp_fused__softmax_mul_sum_165(c_void_p(buf468.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()))
    del buf463
    del buf468
    # Source Nodes: [mul_31, out_378, out_383], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf471 = extern_kernels.convolution(buf470, arg498_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf471, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    del arg498_1
    del buf470
    buf472 = buf458; del buf458  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_166(c_void_p(buf472.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(arg920_1.data_ptr()), c_void_p(arg921_1.data_ptr()), c_void_p(arg499_1.data_ptr()), c_void_p(arg500_1.data_ptr()))
    del arg499_1
    del arg500_1
    del arg920_1
    del arg921_1
    del buf471
    # Source Nodes: [out_387], Original ATen: [aten.convolution]
    buf473 = extern_kernels.convolution(buf472, arg501_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf473, (8, 512, 8, 8), (32768, 1, 4096, 512))
    del arg501_1
    buf474 = buf473; del buf473  # reuse
    buf475 = buf461; del buf461  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_167(c_void_p(buf474.data_ptr()), c_void_p(arg923_1.data_ptr()), c_void_p(arg924_1.data_ptr()), c_void_p(arg502_1.data_ptr()), c_void_p(arg503_1.data_ptr()), c_void_p(arg504_1.data_ptr()), c_void_p(buf475.data_ptr()))
    del arg502_1
    del arg503_1
    del arg504_1
    del arg923_1
    del arg924_1
    # Source Nodes: [out_388, out_389, x_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf476 = extern_kernels.convolution(buf474, buf475, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf476, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
    del buf475
    buf477 = buf476; del buf476  # reuse
    buf478 = reinterpret_tensor(buf465, (8, 512, 1, 1), (512, 1, 4096, 4096), 0); del buf465  # reuse
    buf479 = reinterpret_tensor(buf478, (8, 512, 1, 1), (512, 1, 512, 512), 0); del buf478  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_168(c_void_p(buf477.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(arg926_1.data_ptr()), c_void_p(arg927_1.data_ptr()), c_void_p(arg505_1.data_ptr()), c_void_p(arg506_1.data_ptr()))
    del arg505_1
    del arg506_1
    del arg926_1
    del arg927_1
    # Source Nodes: [x_gap_160, x_gap_161, x_gap_162], Original ATen: [aten.convolution, aten.mean, aten.sum]
    buf480 = extern_kernels.convolution(buf479, arg507_1, arg508_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf480, (8, 256, 1, 1), (256, 1, 256, 256))
    del arg507_1
    del arg508_1
    del buf479
    buf481 = buf480; del buf480  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_169(c_void_p(buf481.data_ptr()), c_void_p(arg929_1.data_ptr()), c_void_p(arg930_1.data_ptr()), c_void_p(arg509_1.data_ptr()), c_void_p(arg510_1.data_ptr()))
    del arg509_1
    del arg510_1
    del arg929_1
    del arg930_1
    # Source Nodes: [x_attn_64, x_gap_163, x_gap_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf482 = extern_kernels.convolution(buf481, arg511_1, arg512_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf482, (8, 1024, 1, 1), (1024, 1, 1024, 1024))
    del arg511_1
    del arg512_1
    del buf481
    buf483 = buf469; del buf469  # reuse
    buf484 = buf474; del buf474  # reuse
    cpp_fused__softmax_mul_sum_170(c_void_p(buf482.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf484.data_ptr()))
    del buf477
    del buf482
    del buf483
    # Source Nodes: [mul_32, out_390, out_395], Original ATen: [aten.convolution, aten.mul, aten.sum]
    buf485 = extern_kernels.convolution(buf484, arg513_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf485, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    del arg513_1
    del buf484
    buf486 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cpu', dtype=torch.float32)
    buf487 = reinterpret_tensor(buf486, (8, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf486  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_171(c_void_p(buf487.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(arg932_1.data_ptr()), c_void_p(arg933_1.data_ptr()), c_void_p(arg514_1.data_ptr()), c_void_p(arg515_1.data_ptr()), c_void_p(buf472.data_ptr()))
    del arg514_1
    del arg515_1
    del arg932_1
    del arg933_1
    del buf472
    del buf485
    buf488 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_276], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg517_1, reinterpret_tensor(buf487, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg516_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf488)
    del arg516_1
    del arg517_1
    return (buf488, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
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
    arg24_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg364_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg365_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg367_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg368_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg369_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg370_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg371_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg372_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg373_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg374_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg375_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg376_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg377_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg378_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg379_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg380_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg381_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg382_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg383_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg384_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg385_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg386_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg387_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg388_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg389_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg390_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg391_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg392_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg393_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg394_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg395_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg396_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg397_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg398_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg399_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg400_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg401_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg402_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg403_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg404_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg405_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg406_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg407_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg408_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg409_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg410_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg411_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg412_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg413_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg414_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg415_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg416_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg417_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg418_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg419_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg420_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg421_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg422_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg423_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg424_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg425_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg426_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg427_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg428_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg429_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg430_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg431_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg432_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg433_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg434_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg435_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg436_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg437_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg438_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg439_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg440_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg441_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg442_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg443_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg444_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg445_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg446_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg447_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg448_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg449_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg450_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg451_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg452_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg453_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg454_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg455_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg456_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg457_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg458_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg459_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg460_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg461_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg462_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg463_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg464_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg465_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg466_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg467_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg468_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg469_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg470_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg471_1 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg472_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg473_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg474_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg475_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg476_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg477_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg478_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg479_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg480_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg481_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg482_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg483_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg484_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg485_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg486_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg487_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg488_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg489_1 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg490_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg491_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg492_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg493_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg494_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg495_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg496_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg497_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg498_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg499_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg500_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg501_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg502_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg503_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg504_1 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg505_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg506_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg507_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg508_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg509_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg510_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg511_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg512_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg513_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg514_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg515_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg516_1 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg517_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg518_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg519_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg520_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg521_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg522_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg523_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg524_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg525_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg526_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg527_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg528_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg529_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg530_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg531_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg532_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg533_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg534_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg535_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg536_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg537_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg538_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg539_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg540_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg541_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg542_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg543_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg544_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg545_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg546_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg547_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg548_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg549_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg550_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg551_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg552_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg553_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg554_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg555_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg556_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg557_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg558_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg559_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg560_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg561_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg562_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg563_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg564_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg565_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg566_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg567_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg568_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg569_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg570_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg571_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg572_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg573_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg574_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg575_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg576_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg577_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg578_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg579_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg580_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg581_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg582_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg583_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg584_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg585_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg586_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg587_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg588_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg589_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg590_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg591_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg592_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg593_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg594_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg595_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg596_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg597_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg598_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg599_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg600_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg601_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg602_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg603_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg604_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg605_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg606_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg607_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg608_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg609_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg610_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg611_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg612_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg613_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg614_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg615_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg616_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg617_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg618_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg619_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg620_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg621_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg622_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg623_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg624_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg625_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg626_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg627_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg628_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg629_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg630_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg631_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg632_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg633_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg634_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg635_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg636_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg637_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg638_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg639_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg640_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg641_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg642_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg643_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg644_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg645_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg646_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg647_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg648_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg649_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg650_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg651_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg652_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg653_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg654_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg655_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg656_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg657_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg658_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg659_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg660_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg661_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg662_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg663_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg664_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg665_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg666_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg667_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg668_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg669_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg670_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg671_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg672_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg673_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg674_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg675_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg676_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg677_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg678_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg679_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg680_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg681_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg682_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg683_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg684_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg685_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg686_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg687_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg688_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg689_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg690_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg691_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg692_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg693_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg694_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg695_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg696_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg697_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg698_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg699_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg700_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg701_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg702_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg703_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg704_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg705_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg706_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg707_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg708_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg709_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg710_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg711_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg712_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg713_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg714_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg715_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg716_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg717_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg718_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg719_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg720_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg721_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg722_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg723_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg724_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg725_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg726_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg727_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg728_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg729_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg730_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg731_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg732_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg733_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg734_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg735_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg736_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg737_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg738_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg739_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg740_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg741_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg742_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg743_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg744_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg745_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg746_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg747_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg748_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg749_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg750_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg751_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg752_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg753_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg754_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg755_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg756_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg757_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg758_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg759_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg760_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg761_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg762_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg763_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg764_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg765_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg766_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg767_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg768_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg769_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg770_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg771_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg772_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg773_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg774_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg775_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg776_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg777_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg778_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg779_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg780_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg781_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg782_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg783_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg784_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg785_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg786_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg787_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg788_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg789_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg790_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg791_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg792_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg793_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg794_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg795_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg796_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg797_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg798_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg799_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg800_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg801_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg802_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg803_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg804_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg805_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg806_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg807_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg808_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg809_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg810_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg811_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg812_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg813_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg814_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg815_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg816_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg817_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg818_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg819_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg820_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg821_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg822_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg823_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg824_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg825_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg826_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg827_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg828_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg829_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg830_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg831_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg832_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg833_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg834_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg835_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg836_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg837_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg838_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg839_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg840_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg841_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg842_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg843_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg844_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg845_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg846_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg847_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg848_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg849_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg850_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg851_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg852_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg853_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg854_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg855_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg856_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg857_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg858_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg859_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg860_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg861_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg862_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg863_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg864_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg865_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg866_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg867_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg868_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg869_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg870_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg871_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg872_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg873_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg874_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg875_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg876_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg877_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg878_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg879_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg880_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg881_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg882_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg883_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg884_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg885_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg886_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg887_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg888_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg889_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg890_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg891_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg892_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg893_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg894_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg895_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg896_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg897_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg898_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg899_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg900_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg901_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg902_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg903_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg904_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg905_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg906_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg907_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg908_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg909_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg910_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg911_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg912_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg913_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg914_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg915_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg916_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg917_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg918_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg919_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg920_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg921_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg922_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg923_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg924_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg925_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg926_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg927_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg928_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg929_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg930_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg931_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg932_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg933_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg934_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg935_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('resnest101e', benchmark_compiled_module)
