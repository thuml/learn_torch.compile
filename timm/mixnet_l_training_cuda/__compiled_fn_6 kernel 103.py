
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_mul_native_batch_norm_backward_102', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 336
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp29 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp33 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        r3 = rindex
        tmp23 = tl.load(in_ptr3 + (x0 + (336*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr4 + (x0 + (336*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 112, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r1 + (784*x0) + (87808*r2)), rmask & tmp4 & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp0 >= tmp3
        tmp9 = tl.full([1, 1], 224, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tmp8 & tmp10
        tmp12 = tl.load(in_ptr1 + ((-87808) + r1 + (784*x0) + (87808*r2)), rmask & tmp11 & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp11, tmp12, tmp13)
        tmp15 = tmp0 >= tmp9
        tmp16 = tl.full([1, 1], 336, tl.int64)
        tmp17 = tmp0 < tmp16
        tmp18 = tl.load(in_ptr2 + ((-175616) + r1 + (784*x0) + (87808*r2)), rmask & tmp15 & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
        tmp20 = tl.where(tmp15, tmp18, tmp19)
        tmp21 = tl.where(tmp11, tmp14, tmp20)
        tmp22 = tl.where(tmp4, tmp7, tmp21)
        tmp24 = tmp22 * tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
        tmp30 = tmp28 - tmp29
        tmp31 = tmp24 * tmp30
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(rmask & xmask, tmp34, _tmp33)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp26, xmask)
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp33, xmask)
    tmp35 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tmp33 * tmp35
    tl.store(out_ptr2 + (x0), tmp36, xmask)
