
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_constant_pad_nd_native_batch_norm_backward_threshold_backward_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 47952
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 111
    x1 = (xindex // 111)
    _tmp37 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x0)
        tmp1 = tl.full([1, 1], 14112, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((1764*x1) + (762048*(((r2 + (128*x0)) // 1764) % 8)) + ((r2 + (128*x0)) % 1764)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = ((r2 + (128*x0)) // 42) % 42
        tmp5 = tl.full([1, 1], 43, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = (r2 + (128*x0)) % 42
        tmp8 = tmp7 < tmp5
        tmp9 = tmp6 & tmp8
        tmp10 = tmp9 & tmp2
        tmp11 = tl.load(in_ptr1 + (x1 + (432*((r2 + (128*x0)) % 42)) + (18576*(((r2 + (128*x0)) // 42) % 42)) + (798768*(((r2 + (128*x0)) // 1764) % 8))), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp10, tmp11, tmp12)
        tmp14 = tmp3 + tmp13
        tmp15 = tl.load(in_ptr2 + (x1 + (432*((r2 + (128*x0)) % 14112))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp16 = 2 + (((r2 + (128*x0)) // 42) % 42)
        tmp17 = tl.full([1, 1], 0, tl.int64)
        tmp18 = tmp16 >= tmp17
        tmp19 = tl.full([1, 1], 47, tl.int64)
        tmp20 = tmp16 < tmp19
        tmp21 = 2 + ((r2 + (128*x0)) % 42)
        tmp22 = tmp21 >= tmp17
        tmp23 = tmp21 < tmp19
        tmp24 = tmp18 & tmp20
        tmp25 = tmp24 & tmp22
        tmp26 = tmp25 & tmp23
        tmp27 = tmp26 & tmp2
        tmp28 = tl.load(in_ptr3 + (96 + (47*(((r2 + (128*x0)) // 42) % 42)) + (2209*x1) + (954288*(((r2 + (128*x0)) // 1764) % 8)) + ((r2 + (128*x0)) % 42)), rmask & tmp27 & xmask, eviction_policy='evict_last', other=0.0)
        tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
        tmp30 = tl.where(tmp27, tmp28, tmp29)
        tmp31 = 0.0
        tmp32 = tl.where(tmp15, tmp31, tmp30)
        tmp33 = tmp14 + tmp32
        tmp34 = tl.full(tmp33.shape, 0, tmp33.dtype)
        tmp35 = tl.where(tmp2, tmp33, tmp34)
        tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
        tmp38 = _tmp37 + tmp36
        _tmp37 = tl.where(rmask & xmask, tmp38, _tmp37)
    tmp37 = tl.sum(_tmp37, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp37, xmask)
