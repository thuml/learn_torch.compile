
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_27', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp20 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp27 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = (rindex // 784)
        r4 = rindex % 784
        r1 = rindex % 28
        r2 = (rindex // 28) % 28
        tmp0 = tl.load(in_ptr0 + (r4 + (784*x0) + (401408*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((14*(tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(14, 1 + (r2 // 2)))))) + (14*(tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(14, 1 + (r2 // 2))))) >= 0, 0, 14))) + (196*x0) + (100352*r3) + (tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(14, 1 + (r1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(14, 1 + (r1 // 2))))) >= 0, 0, 14))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (r4 + (784*x0) + (401408*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r4 + (784*x0) + (401408*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr5 + (r4 + (784*x0) + (401408*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tmp3 / 4
        tmp5 = tl.math.max(0, (r2 // 2))
        tmp6 = tl.math.min(14, 1 + (r2 // 2))
        tmp7 = tmp5 < tmp6
        tmp8 = tl.math.max(0, (r1 // 2))
        tmp9 = tl.math.min(14, 1 + (r1 // 2))
        tmp10 = tmp8 < tmp9
        tmp11 = tmp7 & tmp10
        tmp12 = tl.where(tmp11, tmp4, tmp1)
        tmp14 = tmp12 + tmp13
        tmp15 = tl.where(tmp2, tmp1, tmp14)
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
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp17, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tmp33 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp24 * tmp36
    tmp39 = tmp38 + tmp34
    tmp40 = tl.math.rsqrt(tmp39)
    tmp41 = tmp31 * tmp40
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp37, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp41, xmask)
