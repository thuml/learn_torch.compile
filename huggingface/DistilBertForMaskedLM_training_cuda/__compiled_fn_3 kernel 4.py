
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_lift_fresh_masked_fill_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp2 = -3.4028234663852886e+38
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.max2(tmp6, 1)[:, None]
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tmp9 / tmp13
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp14, rmask & xmask)
