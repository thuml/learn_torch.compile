
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (200704*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (x0 + (2048*(r2 // 49)) + (4096*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (2048*r2) + (200704*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (2048*r2) + (200704*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr5 + ((49*x0) + (100352*(r2 // 49)) + (200704*x1) + (r2 % 49)), rmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr6 + (x0 + (2048*r2) + (200704*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 49.0
        tmp3 = tmp1 / tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp5 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
        tmp17 = tmp16 <= tmp4
        tmp19 = tmp5 + tmp18
        tmp20 = tl.where(tmp17, tmp4, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
        tmp26 = tmp24 - tmp25
        tmp27 = tmp20 * tmp26
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask, tmp30, _tmp29)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp14, None)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp22, None)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr3 + (x3), tmp29, None)
