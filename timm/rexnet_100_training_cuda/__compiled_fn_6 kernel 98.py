
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_slice_backward_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 106
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp26 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        r3 = rindex
        tmp25 = tl.load(in_ptr3 + (x0 + (106*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 95, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + (r1 + (196*x0) + (25088*r2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (196*x0) + (22932*r2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.load(in_ptr2 + (r1 + (196*x0) + (20776*r2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = 0.0
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tmp0 < tmp1
        tmp13 = tl.load(in_ptr0 + (r1 + (196*x0) + (25088*r2)), rmask & tmp12 & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr1 + (r1 + (196*x0) + (22932*r2)), rmask & tmp12 & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tl.load(in_ptr2 + (r1 + (196*x0) + (20776*r2)), rmask & tmp12 & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tmp15 + tmp16
        tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
        tmp19 = tl.where(tmp12, tmp17, tmp18)
        tmp20 = tl.where(tmp12, tmp19, tmp10)
        tmp21 = tmp11 + tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
        tmp27 = tmp25 - tmp26
        tmp28 = tmp21 * tmp27
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask & xmask, tmp31, _tmp30)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp23, xmask)
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp30, xmask)
    tmp32 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tmp30 * tmp32
    tl.store(out_ptr2 + (x0), tmp33, xmask)
