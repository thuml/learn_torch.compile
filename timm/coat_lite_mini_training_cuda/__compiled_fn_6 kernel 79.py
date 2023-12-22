
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_native_layer_norm_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4160
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 320)
    x0 = xindex % 320
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp32 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (320 + x0 + (320*((r2 + (121*x1)) % 196)) + (63040*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp9 = 1 + ((r2 + (121*x1)) % 196)
        tmp10 = tl.full([1, 1], 1, tl.int64)
        tmp11 = tmp9 >= tmp10
        tmp12 = tmp11 & tmp2
        tmp13 = tl.load(in_ptr0 + (320 + x0 + (320*((r2 + (121*x1)) % 196)) + (63040*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp12 & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr1 + ((196*x0) + (62720*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp12 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp12, tmp15, tmp16)
        tmp18 = 0.0
        tmp19 = tl.where(tmp11, tmp17, tmp18)
        tmp20 = tmp9 < tmp10
        tmp21 = tmp20 & tmp2
        tmp22 = tl.load(in_ptr0 + (x0 + (63040*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp21 & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
        tmp24 = tl.where(tmp21, tmp22, tmp23)
        tmp25 = tl.where(tmp20, tmp24, tmp18)
        tmp26 = tmp19 + tmp25
        tmp27 = tl.load(in_ptr2 + (x0 + (320*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tmp26 * tmp27
        tmp29 = tl.full(tmp28.shape, 0, tmp28.dtype)
        tmp30 = tl.where(tmp2, tmp28, tmp29)
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(rmask & xmask, tmp33, _tmp32)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp32, xmask)
