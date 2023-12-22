
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x2 = (xindex // 3136)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp9 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr2 + (r3 + (256*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x1
        tmp1 = tl.full([1, 1], 57, tl.int64)
        tmp2 = tmp0 < tmp1
        tmp3 = x0
        tmp4 = tmp3 < tmp1
        tmp5 = tmp2 & tmp4
        tmp6 = tl.load(in_ptr0 + (r3 + (256*x0) + (14592*x1) + (831744*x2)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp10 = tmp8 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp15 = tmp10 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp19 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp29 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp34 = tl.load(in_ptr2 + (r3 + (256*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = x1
        tmp21 = tl.full([1, 1], 57, tl.int64)
        tmp22 = tmp20 < tmp21
        tmp23 = x0
        tmp24 = tmp23 < tmp21
        tmp25 = tmp22 & tmp24
        tmp26 = tl.load(in_ptr0 + (r3 + (256*x0) + (14592*x1) + (831744*x2)), rmask & tmp25 & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
        tmp28 = tl.where(tmp25, tmp26, tmp27)
        tmp30 = tmp28 * tmp29
        tmp31 = 256.0
        tmp32 = tmp30 * tmp31
        tmp33 = tmp32 - tmp12
        tmp35 = tmp34 * tmp17
        tmp36 = tmp33 - tmp35
        tmp37 = tmp19 * tmp36
        tl.store(out_ptr2 + (r3 + (256*x4)), tmp37, rmask & xmask)
