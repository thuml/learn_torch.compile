
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


cpp_fused__log_softmax_clone_copy_fill_lift_fresh_new_zeros_nll_loss_forward_select_scatter_slice_scatter_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       long* out_ptr3,
                       long* out_ptr4,
                       float* out_ptr6)
{
    auto out_ptr5 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256008L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256008L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256008L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256008L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256008L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256008L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = std::log(tmp4);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp3 - tmp6;
                    tmp7.store(out_ptr2 + static_cast<long>(x1 + (256008L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                {
                    long tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<int>(x0);
                        auto tmp1 = static_cast<int>(127);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp3 = c10::convert<long>(x0);
                        auto tmp4 = static_cast<long>(127);
                        auto tmp5 = tmp3 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>(1L + x0)];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0);
                        auto tmp9 = static_cast<long>(0);
                        auto tmp10 = tmp5 ? tmp8 : tmp9;
                        auto tmp11 = static_cast<long>(1);
                        auto tmp12 = tmp2 ? tmp11 : tmp10;
                        auto tmp13 = static_cast<long>(-100);
                        auto tmp14 = tmp12 != tmp13;
                        auto tmp15 = c10::convert<long>(tmp14);
                        auto tmp16 = tmp14 ? tmp12 : tmp9;
                        auto tmp17 = decltype(tmp16)(tmp16 + 256008);
                        auto tmp18 = tmp16 < 0;
                        auto tmp19 = tmp18 ? tmp17 : tmp16;
                        TORCH_CHECK((0 <= tmp19) & (tmp19 < 256008L), "index out of bounds: 0 <= tmp19 < 256008L")
                        auto tmp20 = out_ptr2[static_cast<long>(tmp19 + (256008L*x0))];
                        auto tmp21 = decltype(tmp20)(-tmp20);
                        auto tmp22 = static_cast<float>(0.0);
                        auto tmp23 = tmp14 ? tmp21 : tmp22;
                        out_ptr3[static_cast<long>(x0)] = tmp12;
                        tmp_acc0 = tmp_acc0 + tmp15;
                        tmp_acc1 = tmp_acc1 + tmp23;
                    }
                    out_ptr4[static_cast<long>(0L)] = tmp_acc0;
                    out_ptr5[static_cast<long>(0L)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = out_ptr4[static_cast<long>(0L)];
                auto tmp2 = out_ptr5[static_cast<long>(0L)];
                auto tmp1 = c10::convert<float>(tmp0);
                auto tmp3 = tmp2 / tmp1;
                out_ptr6[static_cast<long>(0L)] = tmp1;
                in_out_ptr0[static_cast<long>(0L)] = tmp3;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (256008, 1024), (1024, 1))
    assert_size_stride(primals_2, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(primals_3, (1, 128), (128, 1))
    buf0 = empty((128, 256008), device='cpu', dtype=torch.float32)
    # Source Nodes: [logits], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(primals_2, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_1, (1024, 256008), (1, 1024), 0), out=buf0)
    buf2 = empty_strided((128, 1), (1, 128), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((128, 1), (1, 128), device='cpu', dtype=torch.float32)
    buf4 = empty((128, 256008), device='cpu', dtype=torch.float32)
    buf1 = empty((1, 128), device='cpu', dtype=torch.int64)
    buf5 = empty((), device='cpu', dtype=torch.int64)
    buf7 = empty((), device='cpu', dtype=torch.float32)
    buf6 = empty((), device='cpu', dtype=torch.float32)
    buf8 = buf7; del buf7  # reuse
    cpp_fused__log_softmax_clone_copy_fill_lift_fresh_new_zeros_nll_loss_forward_select_scatter_slice_scatter_0(c_void_p(buf8.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del buf2
    del buf3
    del buf5
    del primals_3
    return (buf8, reinterpret_tensor(buf0, (1, 128, 256008), (32769024, 256008, 1), 0), reinterpret_tensor(primals_2, (128, 1024), (1024, 1), 0), buf1, buf4, buf6, reinterpret_tensor(primals_1, (256008, 1024), (1024, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((256008, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('XGLMForCausalLM', benchmark_compiled_module)
