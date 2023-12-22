
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_41', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr4 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr5 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp12 = 0.0
        tmp13 = tmp11 <= tmp12
        tmp15 = tmp0 + tmp14
        tmp16 = tl.where(tmp13, tmp12, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tmp22 = tmp20 - tmp21
        tmp23 = tmp16 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp18, xmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tmp27 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp9 * tmp30
    tmp33 = tmp32 + tmp28
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp25 * tmp34
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp31, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp35, xmask)
