
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_51', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 28
    x1 = (xindex // 28) % 28
    x2 = (xindex // 784)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (192*(tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2)))))) + (192*(tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) >= 0, 0, 14))) + (2688*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2)))))) + (2688*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2))))) >= 0, 0, 14))) + (37632*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr1 + (r3 + (192*x5)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr2 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_ptr3 + (r3 + (192*x5)), rmask & xmask, other=0.0)
    tmp25 = tl.load(in_out_ptr0 + (r3 + (192*x5)), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr4 + (x5), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 / 4
    tmp2 = tl.math.max(0, (x1 // 2))
    tmp3 = tl.math.min(14, 1 + (x1 // 2))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (x0 // 2))
    tmp6 = tl.math.min(14, 1 + (x0 // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = tmp14 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp27 = 192.0
    tmp28 = tmp14 * tmp27
    tmp29 = tmp28 - tmp18
    tmp30 = tmp19 * tmp24
    tmp31 = tmp29 - tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = tmp25 + tmp32
    tl.store(in_out_ptr0 + (r3 + (192*x5)), tmp33, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (192*x5)), tmp33, rmask & xmask)
