
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_slice_backward_111', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 95
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp30 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        r3 = rindex
        tmp29 = tl.load(in_ptr4 + (x0 + (95*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 84, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + (r1 + (196*x0) + (25088*r2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (196*x0) + (22932*r2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.load(in_ptr2 + (r1 + (196*x0) + (20776*r2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.load(in_ptr3 + (r1 + (196*x0) + (18620*r2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = 0.0
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tmp0 < tmp1
        tmp15 = tl.load(in_ptr0 + (r1 + (196*x0) + (25088*r2)), rmask & tmp14 & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr1 + (r1 + (196*x0) + (22932*r2)), rmask & tmp14 & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tmp15 + tmp16
        tmp18 = tl.load(in_ptr2 + (r1 + (196*x0) + (20776*r2)), rmask & tmp14 & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tmp17 + tmp18
        tmp20 = tl.load(in_ptr3 + (r1 + (196*x0) + (18620*r2)), rmask & tmp14 & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tmp19 + tmp20
        tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
        tmp23 = tl.where(tmp14, tmp21, tmp22)
        tmp24 = tl.where(tmp14, tmp23, tmp12)
        tmp25 = tmp13 + tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
        tmp31 = tmp29 - tmp30
        tmp32 = tmp25 * tmp31
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask & xmask, tmp35, _tmp34)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp27, xmask)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp34, xmask)
    tmp36 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp37 = tmp34 * tmp36
    tl.store(out_ptr2 + (x0), tmp37, xmask)
