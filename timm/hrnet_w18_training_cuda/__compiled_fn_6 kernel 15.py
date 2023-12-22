
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: 'i32', 21: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(20, 21))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp20 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr4 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp19 = tl.load(in_ptr5 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr7 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp15 = tl.where(tmp14, tmp2, tmp1)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp15 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
        tmp28 = tmp26 - tmp27
        tmp29 = tmp15 * tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask & xmask, tmp32, _tmp31)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp12, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp17, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr3 + (x0), tmp24, xmask)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tl.store(out_ptr4 + (x0), tmp31, xmask)
    tmp33 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp34 = tmp24 * tmp33
    tmp36 = tmp31 * tmp35
    tmp38 = tmp12 * tmp37
    tl.store(out_ptr5 + (x0), tmp34, xmask)
    tl.store(out_ptr6 + (x0), tmp36, xmask)
    tl.store(out_ptr7 + (x0), tmp38, xmask)
