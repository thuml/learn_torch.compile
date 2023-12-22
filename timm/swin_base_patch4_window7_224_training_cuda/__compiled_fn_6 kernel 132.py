
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_131', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 56
    x1 = (xindex // 56) % 56
    x2 = (xindex // 3136)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*(x0 % 7)) + (896*(x1 % 7)) + (6272*(x0 // 7)) + (50176*(x1 // 7)) + (401408*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp7 = tmp2 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp12 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp11 = tl.load(in_out_ptr0 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr0 + (r3 + (128*(x0 % 7)) + (896*(x1 % 7)) + (6272*(x0 // 7)) + (50176*(x1 // 7)) + (401408*x2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr2 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 * tmp14
        tmp16 = 128.0
        tmp17 = tmp15 * tmp16
        tmp18 = tmp17 - tmp4
        tmp20 = tmp19 * tmp9
        tmp21 = tmp18 - tmp20
        tmp22 = tmp12 * tmp21
        tmp23 = tmp11 + tmp22
        tmp25 = tmp23 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
        tl.store(in_out_ptr0 + (r3 + (128*x4)), tmp23, rmask & xmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    _tmp35 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr5 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tmp29 * tmp30
        tmp33 = tmp31 * tmp32
        tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
        tmp36 = _tmp35 + tmp34
        _tmp35 = tl.where(rmask & xmask, tmp36, _tmp35)
    tmp35 = tl.sum(_tmp35, 1)[:, None]
    tmp37 = tl.load(in_ptr6 + (x4), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp38 = tl.load(in_out_ptr0 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp39 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp44 = tl.load(in_ptr5 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp40 = tmp38 * tmp39
        tmp41 = 128.0
        tmp42 = tmp40 * tmp41
        tmp43 = tmp42 - tmp27
        tmp45 = tmp44 * tmp35
        tmp46 = tmp43 - tmp45
        tmp47 = tmp37 * tmp46
        tl.store(out_ptr4 + (r3 + (128*x4)), tmp47, rmask & xmask)
