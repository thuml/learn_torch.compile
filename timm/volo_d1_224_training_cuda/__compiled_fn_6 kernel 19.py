
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_slice_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = (r2 + (122*x1)) % 197
        tmp4 = tl.full([1, 1], 1, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = tmp5 & tmp2
        tmp7 = tl.load(in_ptr0 + (x0 + (384*(((r2 + (122*x1)) // 197) % 8))), rmask & tmp6 & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp10 = 0.0
        tmp11 = tl.where(tmp5, tmp9, tmp10)
        tmp12 = tl.load(in_ptr1 + (x0 + (384*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp14 = tl.load(in_ptr2 + (x0 + (384*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr3 + ((r2 + (122*x1)) % 1576), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp14 - tmp15
        tmp17 = tl.load(in_ptr4 + ((r2 + (122*x1)) % 1576), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp13 * tmp18
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
        tmp25 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp26 = tl.where(tmp2, tmp13, tmp25)
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, xmask)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp28, xmask)
