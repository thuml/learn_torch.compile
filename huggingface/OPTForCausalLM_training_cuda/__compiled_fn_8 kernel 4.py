
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*i1', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_clone_detach_eq_lift_fresh_lt_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2048
    _tmp6 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = -3.4028234663852886e+38
        tmp4 = triton_helpers.maximum(tmp2, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = triton_helpers.maximum(_tmp6, tmp5)
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = triton_helpers.max2(_tmp6, 1)[:, None]
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp8 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr1 + (r2 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 + tmp9
        tmp11 = -3.4028234663852886e+38
        tmp12 = triton_helpers.maximum(tmp10, tmp11)
        tmp13 = tmp12 - tmp6
        tmp14 = tl.exp(tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp18 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr1 + (r2 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp18 + tmp19
        tmp21 = -3.4028234663852886e+38
        tmp22 = triton_helpers.maximum(tmp20, tmp21)
        tmp23 = tmp22 - tmp6
        tmp24 = tl.exp(tmp23)
        tmp25 = tmp24 / tmp16
        tmp26 = tmp20 == tmp21
        tmp27 = tmp20 < tmp21
        tl.store(out_ptr2 + (r2 + (2048*x3)), tmp25, rmask)
        tl.store(out_ptr3 + (r2 + (2048*x3)), tmp25, rmask)
        tl.store(out_ptr4 + (r2 + (2048*x3)), tmp26, rmask)
        tl.store(out_ptr5 + (r2 + (2048*x3)), tmp27, rmask)
