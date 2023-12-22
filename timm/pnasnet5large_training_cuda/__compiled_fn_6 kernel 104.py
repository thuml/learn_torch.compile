
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_constant_pad_nd_native_batch_norm_backward_threshold_backward_103', 'mutated_arg_names': []}
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
    _tmp44 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 14112, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (432*((r2 + (128*x1)) % 14112))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = ((r2 + (128*x1)) // 42) % 42
        tmp5 = tl.full([1, 1], 43, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = (r2 + (128*x1)) % 42
        tmp8 = tmp7 < tmp5
        tmp9 = tmp6 & tmp8
        tmp10 = tmp9 & tmp2
        tmp11 = tl.load(in_ptr1 + ((43*(((r2 + (128*x1)) // 42) % 42)) + (1849*x0) + (798768*(((r2 + (128*x1)) // 1764) % 8)) + ((r2 + (128*x1)) % 42)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp10, tmp11, tmp12)
        tmp14 = 0.0
        tmp15 = tl.where(tmp3, tmp14, tmp13)
        tmp16 = tl.load(in_ptr2 + (x0 + (432*((r2 + (128*x1)) % 42)) + (18576*(((r2 + (128*x1)) // 42) % 42)) + (798768*(((r2 + (128*x1)) // 1764) % 8))), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
        tmp18 = tl.where(tmp10, tmp16, tmp17)
        tmp19 = tmp15 + tmp18
        tmp20 = 1 + (((r2 + (128*x1)) // 42) % 42)
        tmp21 = tl.full([1, 1], 0, tl.int64)
        tmp22 = tmp20 >= tmp21
        tmp23 = tl.full([1, 1], 45, tl.int64)
        tmp24 = tmp20 < tmp23
        tmp25 = 1 + ((r2 + (128*x1)) % 42)
        tmp26 = tmp25 >= tmp21
        tmp27 = tmp25 < tmp23
        tmp28 = tmp22 & tmp24
        tmp29 = tmp28 & tmp26
        tmp30 = tmp29 & tmp27
        tmp31 = tmp30 & tmp2
        tmp32 = tl.load(in_ptr3 + (46 + (45*(((r2 + (128*x1)) // 42) % 42)) + (2025*x0) + (874800*(((r2 + (128*x1)) // 1764) % 8)) + ((r2 + (128*x1)) % 42)), rmask & tmp31 & xmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
        tmp34 = tl.where(tmp31, tmp32, tmp33)
        tmp35 = tl.where(tmp3, tmp14, tmp34)
        tmp36 = tmp19 + tmp35
        tmp37 = tl.load(in_ptr4 + (x0 + (432*((r2 + (128*x1)) % 14112))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp38 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp39 = tmp37 - tmp38
        tmp40 = tmp36 * tmp39
        tmp41 = tl.full(tmp40.shape, 0, tmp40.dtype)
        tmp42 = tl.where(tmp2, tmp40, tmp41)
        tmp43 = tl.broadcast_to(tmp42, [XBLOCK, RBLOCK])
        tmp45 = _tmp44 + tmp43
        _tmp44 = tl.where(rmask & xmask, tmp45, _tmp44)
    tmp44 = tl.sum(_tmp44, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp44, xmask)
