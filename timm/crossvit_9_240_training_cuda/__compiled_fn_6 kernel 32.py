
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.7071067811865476
    tmp7 = tmp5 * tmp6
    tmp8 = tl.math.erf(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = 0.5
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 * tmp5
    tmp14 = -0.5
    tmp15 = tmp13 * tmp14
    tmp16 = tl.exp(tmp15)
    tmp17 = 0.3989422804014327
    tmp18 = tmp16 * tmp17
    tmp19 = tmp5 * tmp18
    tmp20 = tmp12 + tmp19
    tmp21 = tmp0 * tmp20
    tmp22 = tmp21 * tmp2
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tmp22 * tmp1
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tl.store(out_ptr0 + (x0), tmp26, xmask)
    tl.store(out_ptr1 + (x0), tmp31, xmask)
