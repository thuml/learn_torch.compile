
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_select_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6280
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 785
    x1 = (xindex // 785)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp3 = tl.load(in_ptr0 + (r2 + (768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp12 = tmp7 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp16 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp20 = tl.load(in_ptr0 + (r2 + (768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = x0
        tmp18 = tl.full([1, 1], 0, tl.int32)
        tmp19 = tmp17 == tmp18
        tmp21 = 0.0
        tmp22 = tl.where(tmp19, tmp20, tmp21)
        tmp24 = tmp22 * tmp23
        tmp25 = 768.0
        tmp26 = tmp24 * tmp25
        tmp27 = tmp26 - tmp9
        tmp29 = tmp28 * tmp14
        tmp30 = tmp27 - tmp29
        tmp31 = tmp16 * tmp30
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp31, rmask & xmask)
