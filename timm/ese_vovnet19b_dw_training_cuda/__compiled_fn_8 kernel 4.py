
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_hardsigmoid_backward_mul_sum_1', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x3 = xindex
    r2 = rindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0 + (1024*r2) + (50176*x1)), rmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last').to(tl.int1)
    tmp1 = 49.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp10 = 0.16666666666666666
    tmp11 = tmp8 * tmp10
    tmp12 = 0.0
    tmp13 = tl.where(tmp9, tmp11, tmp12)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp13, None)
