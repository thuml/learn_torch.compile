
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_avg_pool2d_backward_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_61', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    _tmp51 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        r2 = rindex % 48
        r3 = (rindex // 48)
        tmp0 = tl.load(in_ptr0 + (r1 + (2304*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r1 + (2304*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_out_ptr0 + (r1 + (2304*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp29 = tl.load(in_ptr4 + ((24*(tl.math.min(tl.math.max(0, (r3 // 2)), (-1) + (tl.math.min(24, 1 + (r3 // 2)))))) + (24*(tl.where((tl.math.min(tl.math.max(0, (r3 // 2)), (-1) + (tl.math.min(24, 1 + (r3 // 2))))) >= 0, 0, 24))) + (576*x0) + (tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(24, 1 + (r2 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(24, 1 + (r2 // 2))))) >= 0, 0, 24))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 2.0
        tmp5 = tmp3 * tmp4
        tmp8 = tmp5 * tmp7
        tmp9 = 0.2
        tmp10 = tmp8 * tmp9
        tmp12 = tmp10 + tmp11
        tmp13 = 0.7071067811865476
        tmp14 = tmp12 * tmp13
        tmp15 = tl.math.erf(tmp14)
        tmp16 = 1.0
        tmp17 = tmp15 + tmp16
        tmp18 = 0.5
        tmp19 = tmp17 * tmp18
        tmp20 = tmp12 * tmp12
        tmp21 = -0.5
        tmp22 = tmp20 * tmp21
        tmp23 = tl.exp(tmp22)
        tmp24 = 0.3989422804014327
        tmp25 = tmp23 * tmp24
        tmp26 = tmp12 * tmp25
        tmp27 = tmp19 + tmp26
        tmp30 = tmp29 / 4
        tmp31 = tl.math.max(0, (r3 // 2))
        tmp32 = tl.math.min(24, 1 + (r3 // 2))
        tmp33 = tmp31 < tmp32
        tmp34 = tl.math.max(0, (r2 // 2))
        tmp35 = tl.math.min(24, 1 + (r2 // 2))
        tmp36 = tmp34 < tmp35
        tmp37 = tmp33 & tmp36
        tmp38 = 0.0
        tmp39 = tl.where(tmp37, tmp30, tmp38)
        tmp40 = tmp28 + tmp39
        tmp41 = 0.9805806756909201
        tmp42 = tmp40 * tmp41
        tmp43 = 1.7015043497085571
        tmp44 = tmp42 * tmp43
        tmp45 = tmp44 * tmp27
        tmp46 = tmp45 * tmp9
        tmp47 = tmp46 * tmp7
        tmp48 = tmp47 * tmp4
        tmp49 = tmp48 * tmp0
        tmp50 = tl.broadcast_to(tmp49, [XBLOCK, RBLOCK])
        tmp52 = _tmp51 + tmp50
        _tmp51 = tl.where(rmask & xmask, tmp52, _tmp51)
        tl.store(in_out_ptr0 + (r1 + (2304*x0)), tmp45, rmask & xmask)
    tmp51 = tl.sum(_tmp51, 1)[:, None]
    tmp53 = tl.sigmoid(tmp1)
    tmp54 = 1.0
    tmp55 = tmp54 - tmp53
    tmp56 = tmp53 * tmp55
    tmp57 = tmp51 * tmp56
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp57, xmask)
