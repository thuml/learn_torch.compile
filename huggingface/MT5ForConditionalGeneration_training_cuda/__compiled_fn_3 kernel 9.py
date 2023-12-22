
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp1 = (-1)*(tl.math.min(0, r2 + ((-1)*x0)))
    tmp2 = tl.full([1, 1], 16, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tmp1.to(tl.float32)
    tmp5 = 16.0
    tmp6 = tmp4 / tmp5
    tmp7 = tl.log(tmp6)
    tmp8 = 2.0794415416798357
    tmp9 = tmp7 / tmp8
    tmp10 = tmp9 * tmp5
    tmp11 = tmp10.to(tl.int64)
    tmp12 = tmp11 + tmp2
    tmp13 = tl.full([1, 1], 31, tl.int64)
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tmp15 = tl.where(tmp3, tmp1, tmp14)
    tmp16 = tl.full([1, 1], 0, tl.int64)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17 + 32
    tmp19 = tmp17 < 0
    tmp20 = tl.where(tmp19, tmp18, tmp17)
    tl.device_assert((0 <= tmp20) & (tmp20 < 32), "index out of bounds: 0 <= tmp20 < 32")
    tmp21 = tl.load(in_ptr1 + (x1 + (6*tmp20)), xmask, eviction_policy='evict_last')
    tmp22 = r2
    tmp23 = x0
    tmp24 = tmp22 <= tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = 1.0
    tmp27 = tmp26 - tmp25
    tmp28 = -3.4028234663852886e+38
    tmp29 = tmp27 * tmp28
    tmp30 = tmp21 + tmp29
    tmp31 = tmp0 + tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
    tmp34 = tl.where(rmask & xmask, tmp32, float("-inf"))
    tmp35 = triton_helpers.max2(tmp34, 1)[:, None]
    tmp36 = tmp31 - tmp35
    tmp37 = tl.exp(tmp36)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp40 = tl.where(rmask & xmask, tmp38, 0)
    tmp41 = tl.sum(tmp40, 1)[:, None]
    tmp42 = tmp37 / tmp41
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp42, rmask & xmask)
