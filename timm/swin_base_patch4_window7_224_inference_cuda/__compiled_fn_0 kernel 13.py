
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 4
    x2 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r3 + (49*x0) + (2401*(x2 % 64))), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 + 169
    tmp3 = tmp1 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp1)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 169)) | ~rmask, "index out of bounds: 0 <= tmp4 < 169")
    tmp5 = tl.load(in_ptr2 + (x1 + (4*tmp4)), rmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tl.store(out_ptr3 + (r3 + (49*x4)), tmp19, rmask)
