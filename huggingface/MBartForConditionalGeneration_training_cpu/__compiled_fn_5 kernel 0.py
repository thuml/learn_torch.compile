
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


cpp_fused_clone_copy_eq_fill_masked_fill_ne_select_scatter_slice_scatter_sum_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       long* out_ptr0,
                       long* out_ptr1)
{
    {
        {
            long tmp_acc0 = 0;
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<long>(-100);
                auto tmp2 = tmp0 == tmp1;
                auto tmp3 = static_cast<long>(1);
                auto tmp4 = tmp2 ? tmp3 : tmp0;
                auto tmp5 = tmp4 != tmp3;
                auto tmp6 = c10::convert<long>(tmp5);
                tmp_acc0 = tmp_acc0 + tmp6;
            }
            out_ptr0[static_cast<long>(0L)] = tmp_acc0;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
        {
            auto tmp3 = out_ptr0[static_cast<long>(0L)];
            auto tmp22 = in_ptr0[static_cast<long>(x0)];
            auto tmp0 = c10::convert<int>(x0);
            auto tmp1 = static_cast<int>(0);
            auto tmp2 = tmp0 == tmp1;
            auto tmp4 = static_cast<long>(1);
            auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
            auto tmp6 = decltype(tmp5)(tmp5 + 1024);
            auto tmp7 = tmp5 < 0;
            auto tmp8 = tmp7 ? tmp6 : tmp5;
            TORCH_CHECK((0 <= tmp8) & (tmp8 < 1024L), "index out of bounds: 0 <= tmp8 < 1024L")
            auto tmp9 = in_ptr0[static_cast<long>(tmp8)];
            auto tmp10 = static_cast<long>(-100);
            auto tmp11 = tmp9 == tmp10;
            auto tmp12 = tmp11 ? tmp4 : tmp9;
            auto tmp13 = c10::convert<long>(x0);
            auto tmp14 = tmp13 >= tmp4;
            auto tmp15 = [&]
            {
                auto tmp16 = in_ptr0[static_cast<long>((-1L) + x0)];
                auto tmp17 = static_cast<long>(-100);
                auto tmp18 = tmp16 == tmp17;
                auto tmp19 = static_cast<long>(1);
                auto tmp20 = tmp18 ? tmp19 : tmp16;
                return tmp20;
            }
            ;
            auto tmp21 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0);
            auto tmp23 = tmp22 == tmp10;
            auto tmp24 = tmp23 ? tmp4 : tmp22;
            auto tmp25 = tmp14 ? tmp21 : tmp24;
            auto tmp26 = tmp2 ? tmp12 : tmp25;
            out_ptr1[static_cast<long>(x0)] = tmp26;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1024), (1024, 1))
    buf0 = empty((1, ), device='cpu', dtype=torch.int64)
    buf1 = empty((1, 1024), device='cpu', dtype=torch.int64)
    cpp_fused_clone_copy_eq_fill_masked_fill_ne_select_scatter_slice_scatter_sum_0(c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MBartForConditionalGeneration', benchmark_compiled_module)
