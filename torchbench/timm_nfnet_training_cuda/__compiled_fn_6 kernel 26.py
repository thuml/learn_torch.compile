
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_gelu_gelu_backward_mul_sigmoid_sum_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 27
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp44 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (((r1 + (8192*x0)) // 36)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr3 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr4 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp37 = tl.load(in_ptr5 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp38 = tl.load(in_ptr6 + (((r1 + (8192*x0)) // 36)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.2
        tmp2 = tmp0 * tmp1
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 2.0
        tmp8 = tmp6 * tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp14 = 0.9622504486493761
        tmp15 = tmp13 * tmp14
        tmp16 = 1.7015043497085571
        tmp17 = tmp15 * tmp16
        tmp19 = 0.7071067811865476
        tmp20 = tmp18 * tmp19
        tmp21 = tl.math.erf(tmp20)
        tmp22 = 1.0
        tmp23 = tmp21 + tmp22
        tmp24 = 0.5
        tmp25 = tmp23 * tmp24
        tmp26 = tmp18 * tmp18
        tmp27 = -0.5
        tmp28 = tmp26 * tmp27
        tmp29 = tl.exp(tmp28)
        tmp30 = 0.3989422804014327
        tmp31 = tmp29 * tmp30
        tmp32 = tmp18 * tmp31
        tmp33 = tmp25 + tmp32
        tmp34 = tmp17 * tmp33
        tmp35 = tmp0 + tmp34
        tmp36 = tmp35 * tmp1
        tmp39 = tl.sigmoid(tmp38)
        tmp40 = tmp37 * tmp39
        tmp41 = tmp40 * tmp7
        tmp42 = tmp36 * tmp41
        tmp43 = tl.broadcast_to(tmp42, [XBLOCK, RBLOCK])
        tmp45 = _tmp44 + tmp43
        _tmp44 = tl.where(rmask & xmask, tmp45, _tmp44)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tmp44 = tl.sum(_tmp44, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp44, xmask)
