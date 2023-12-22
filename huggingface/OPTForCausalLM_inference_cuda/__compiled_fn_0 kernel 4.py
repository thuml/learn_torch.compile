
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2048
    _tmp10 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = r2
        tmp2 = 1 + x0
        tmp3 = tmp1 < tmp2
        tmp4 = 0.0
        tmp5 = -3.4028234663852886e+38
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tmp0 + tmp6
        tmp8 = triton_helpers.maximum(tmp7, tmp5)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = triton_helpers.maximum(_tmp10, tmp9)
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = triton_helpers.max2(_tmp10, 1)[:, None]
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp12 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = r2
        tmp14 = 1 + x0
        tmp15 = tmp13 < tmp14
        tmp16 = 0.0
        tmp17 = -3.4028234663852886e+38
        tmp18 = tl.where(tmp15, tmp16, tmp17)
        tmp19 = tmp12 + tmp18
        tmp20 = triton_helpers.maximum(tmp19, tmp17)
        tmp21 = tmp20 - tmp10
        tmp22 = tl.exp(tmp21)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask, tmp25, _tmp24)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp26 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp27 = r2
        tmp28 = 1 + x0
        tmp29 = tmp27 < tmp28
        tmp30 = 0.0
        tmp31 = -3.4028234663852886e+38
        tmp32 = tl.where(tmp29, tmp30, tmp31)
        tmp33 = tmp26 + tmp32
        tmp34 = triton_helpers.maximum(tmp33, tmp31)
        tmp35 = tmp34 - tmp10
        tmp36 = tl.exp(tmp35)
        tmp37 = tmp36 / tmp24
        tl.store(out_ptr2 + (r2 + (2048*x3)), tmp37, rmask)
