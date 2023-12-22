
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mul_sum_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 608
    rnumel = 8088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 38
    x1 = (xindex // 38)
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (8088*x0)
        tmp1 = tl.full([1, 1], 307328, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((38416*x1) + (614656*(((r2 + (8088*x0)) // 38416) % 8)) + ((r2 + (8088*x0)) % 38416)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((196*x1) + (3136*(((r2 + (8088*x0)) // 38416) % 8)) + (((r2 + (8088*x0)) // 196) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 / tmp4
        tmp6 = tl.load(in_ptr2 + ((196*x1) + (3136*(((r2 + (8088*x0)) // 38416) % 8)) + (((r2 + (8088*x0)) // 196) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.load(in_ptr3 + ((38416*x1) + (614656*(((r2 + (8088*x0)) // 38416) % 8)) + ((r2 + (8088*x0)) % 38416)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 * tmp8
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp15 = tl.load(in_ptr4 + ((38416*x1) + (614656*(((r2 + (8088*x0)) // 38416) % 8)) + ((r2 + (8088*x0)) % 38416)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp7 * tmp15
        tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
        tmp18 = tl.where(tmp2, tmp16, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
        tmp22 = tl.load(in_ptr5 + ((38416*x1) + (614656*(((r2 + (8088*x0)) // 38416) % 8)) + ((r2 + (8088*x0)) % 38416)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr6 + ((196*x1) + (3136*(((r2 + (8088*x0)) // 38416) % 8)) + (((r2 + (8088*x0)) // 196) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tmp8 * tmp23
        tmp25 = tmp22 - tmp24
        tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
        tmp27 = tl.where(tmp2, tmp25, tmp26)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask & xmask, tmp30, _tmp29)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp20, xmask)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp29, xmask)
