
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 196
    r2 = rindex
    x1 = (xindex // 196)
    x3 = xindex
    tmp16 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr3 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = 1 + x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (320 + r2 + (320*x0) + (63040*x1)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (x0 + (196*r2) + (62720*x1)), rmask & tmp2 & xmask, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (r2 + (63040*x1)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.where(tmp10, tmp13, tmp8)
    tmp15 = tmp9 + tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp25 = 320.0
    tmp26 = tmp17 * tmp25
    tmp28 = tmp26 - tmp27
    tmp29 = tmp18 * tmp23
    tmp30 = tmp28 - tmp29
    tmp31 = tmp24 * tmp30
    tl.store(out_ptr1 + (r2 + (320*x3)), tmp31, rmask & xmask)
