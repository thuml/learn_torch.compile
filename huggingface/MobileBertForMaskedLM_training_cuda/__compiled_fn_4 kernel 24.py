
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr6 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp5 = tmp0 + tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = tmp9 + tmp14
        tmp16 = tmp5 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
        tmp24 = tmp20 * tmp10
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp18, xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr4 + (x0), tmp22, xmask)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr5 + (x0), tmp26, xmask)
