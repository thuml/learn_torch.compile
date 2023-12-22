
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_6', 'mutated_arg_names': []}
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
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = (r2 + (122*x1)) % 197
        tmp4 = tl.full([1, 1], 1, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tmp5 & tmp2
        tmp7 = tl.load(in_ptr0 + ((-384) + x0 + (384*((r2 + (122*x1)) % 197)) + (75264*(((r2 + (122*x1)) // 197) % 8))), rmask & tmp6 & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp10 = 0.0
        tmp11 = tl.where(tmp5, tmp9, tmp10)
        tmp12 = tl.full([1, 1], 0, tl.int32)
        tmp13 = tmp3 == tmp12
        tmp14 = tl.load(in_ptr1 + (x0 + (384*(((r2 + (122*x1)) // 197) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.where(tmp13, tmp14, tmp10)
        tmp16 = tmp11 + tmp15
        tmp17 = tl.load(in_ptr2 + (x0 + (384*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr3 + ((r2 + (122*x1)) % 1576), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tmp17 - tmp18
        tmp20 = tl.load(in_ptr4 + ((r2 + (122*x1)) % 1576), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp19 * tmp20
        tmp22 = tmp16 * tmp21
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
        tmp28 = tl.full(tmp16.shape, 0, tmp16.dtype)
        tmp29 = tl.where(tmp2, tmp16, tmp28)
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask & xmask, tmp32, _tmp31)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp26, xmask)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp31, xmask)
