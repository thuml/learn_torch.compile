
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_sum_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp38 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp42 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp46 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr3 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr5 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr6 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr7 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp32 = tl.load(in_ptr8 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp36 = tl.load(in_ptr9 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp40 = tl.load(in_ptr10 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp44 = tl.load(in_ptr11 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask, tmp19, _tmp18)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask, tmp27, _tmp26)
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask, tmp35, _tmp34)
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(rmask, tmp39, _tmp38)
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK, RBLOCK])
        tmp43 = _tmp42 + tmp41
        _tmp42 = tl.where(rmask, tmp43, _tmp42)
        tmp45 = tl.broadcast_to(tmp44, [XBLOCK, RBLOCK])
        tmp47 = _tmp46 + tmp45
        _tmp46 = tl.where(rmask, tmp47, _tmp46)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    tmp42 = tl.sum(_tmp42, 1)[:, None]
    tmp46 = tl.sum(_tmp46, 1)[:, None]
    tmp48 = tmp2 + tmp6
    tmp49 = tmp48 + tmp10
    tmp50 = tmp49 + tmp14
    tmp51 = tmp50 + tmp18
    tmp52 = tmp51 + tmp22
    tmp53 = tmp52 + tmp30
    tmp54 = tmp53 + tmp38
    tmp55 = tmp54 + tmp46
    tmp56 = tmp55 + tmp26
    tmp57 = tmp56 + tmp34
    tmp58 = tmp57 + tmp42
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp58, None)
