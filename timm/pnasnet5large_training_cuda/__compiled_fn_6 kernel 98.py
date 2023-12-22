
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_constant_pad_nd_native_batch_norm_backward_threshold_backward_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 47952
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 432)
    x0 = xindex % 432
    _tmp41 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 14112, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((1764*x0) + (762048*(((r2 + (128*x1)) // 1764) % 8)) + ((r2 + (128*x1)) % 1764)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = ((r2 + (128*x1)) // 42) % 42
        tmp5 = tl.full([1, 1], 43, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = (r2 + (128*x1)) % 42
        tmp8 = tmp7 < tmp5
        tmp9 = tmp6 & tmp8
        tmp10 = tmp9 & tmp2
        tmp11 = tl.load(in_ptr1 + (x0 + (432*((r2 + (128*x1)) % 42)) + (18576*(((r2 + (128*x1)) // 42) % 42)) + (798768*(((r2 + (128*x1)) // 1764) % 8))), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp10, tmp11, tmp12)
        tmp14 = tmp3 + tmp13
        tmp15 = tl.load(in_ptr2 + (x0 + (432*((r2 + (128*x1)) % 14112))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp16 = 2 + (((r2 + (128*x1)) // 42) % 42)
        tmp17 = tl.full([1, 1], 0, tl.int64)
        tmp18 = tmp16 >= tmp17
        tmp19 = tl.full([1, 1], 47, tl.int64)
        tmp20 = tmp16 < tmp19
        tmp21 = 2 + ((r2 + (128*x1)) % 42)
        tmp22 = tmp21 >= tmp17
        tmp23 = tmp21 < tmp19
        tmp24 = tmp18 & tmp20
        tmp25 = tmp24 & tmp22
        tmp26 = tmp25 & tmp23
        tmp27 = tmp26 & tmp2
        tmp28 = tl.load(in_ptr3 + (96 + (47*(((r2 + (128*x1)) // 42) % 42)) + (2209*x0) + (954288*(((r2 + (128*x1)) // 1764) % 8)) + ((r2 + (128*x1)) % 42)), rmask & tmp27 & xmask, eviction_policy='evict_last', other=0.0)
        tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
        tmp30 = tl.where(tmp27, tmp28, tmp29)
        tmp31 = 0.0
        tmp32 = tl.where(tmp15, tmp31, tmp30)
        tmp33 = tmp14 + tmp32
        tmp34 = tl.load(in_ptr4 + (x0 + (432*((r2 + (128*x1)) % 14112))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp35 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp36 = tmp34 - tmp35
        tmp37 = tmp33 * tmp36
        tmp38 = tl.full(tmp37.shape, 0, tmp37.dtype)
        tmp39 = tl.where(tmp2, tmp37, tmp38)
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(rmask & xmask, tmp42, _tmp41)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp41, xmask)
