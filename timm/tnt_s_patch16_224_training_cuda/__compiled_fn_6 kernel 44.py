
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp33 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = 1 + ((r2 + (121*x1)) % 196)
        tmp4 = tl.full([1, 1], 1, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tmp5 & tmp2
        tmp7 = tl.load(in_ptr0 + (384 + x0 + (384*((r2 + (121*x1)) % 196)) + (75648*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp6 & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp10 = 0.0
        tmp11 = tl.where(tmp5, tmp9, tmp10)
        tmp12 = tmp3 < tmp4
        tmp13 = tmp12 & tmp2
        tmp14 = tl.load(in_ptr0 + (x0 + (75648*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp13 & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
        tmp16 = tl.where(tmp13, tmp14, tmp15)
        tmp17 = tl.where(tmp12, tmp16, tmp10)
        tmp18 = tmp11 + tmp17
        tmp19 = tl.load(in_ptr1 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr2 + ((r2 + (121*x1)) % 1568), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp19 - tmp20
        tmp22 = tl.load(in_ptr3 + ((r2 + (121*x1)) % 1568), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tmp21 * tmp22
        tmp24 = tmp18 * tmp23
        tmp25 = tl.full(tmp24.shape, 0, tmp24.dtype)
        tmp26 = tl.where(tmp2, tmp24, tmp25)
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
        tmp30 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp31 = tl.where(tmp2, tmp18, tmp30)
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(rmask & xmask, tmp34, _tmp33)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp33, xmask)
