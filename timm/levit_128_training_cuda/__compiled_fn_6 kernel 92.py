
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1664
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = (((r2 + (121*x1)) % 196) // 14) % 2
        tmp4 = tl.full([1, 1], 0, tl.int64)
        tmp5 = tmp3 == tmp4
        tmp6 = tmp5 & tmp2
        tmp7 = (((r2 + (121*x1)) % 196) % 14) % 2
        tmp8 = tmp7 == tmp4
        tmp9 = tmp8 & tmp6
        tmp10 = tl.load(in_ptr0 + (x0 + (128*((((r2 + (121*x1)) % 196) % 14) // 2)) + (896*(((r2 + (121*x1)) % 196) // 28)) + (6272*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp9 & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp9, tmp10, tmp11)
        tmp13 = 0.0
        tmp14 = tl.where(tmp8, tmp12, tmp13)
        tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
        tmp16 = tl.where(tmp6, tmp14, tmp15)
        tmp17 = tl.where(tmp5, tmp16, tmp13)
        tmp18 = tl.load(in_ptr1 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tmp17 + tmp18
        tmp20 = tl.load(in_ptr2 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tmp19 + tmp20
        tmp22 = tl.full(tmp21.shape, 0, tmp21.dtype)
        tmp23 = tl.where(tmp2, tmp21, tmp22)
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
        tmp27 = tl.load(in_ptr3 + (x0 + (128*r2) + (15488*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr4 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp29 = tmp27 - tmp28
        tmp30 = tmp21 * tmp29
        tmp31 = tl.full(tmp30.shape, 0, tmp30.dtype)
        tmp32 = tl.where(tmp2, tmp30, tmp31)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask & xmask, tmp35, _tmp34)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp25, xmask)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp34, xmask)
