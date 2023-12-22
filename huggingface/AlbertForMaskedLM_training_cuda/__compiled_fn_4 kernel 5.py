
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_4', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp11 = 1.0
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 * tmp12
    tmp15 = tmp13 - tmp14
    tmp17 = tmp15 * tmp16
    tmp18 = tmp2 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = 128.0
    tmp24 = tmp16 / tmp23
    tmp25 = tmp2 * tmp23
    tmp26 = tmp25 - tmp6
    tmp27 = tmp17 * tmp22
    tmp28 = tmp26 - tmp27
    tmp29 = tmp24 * tmp28
    tmp30 = tmp29 * tmp9
    tmp31 = tmp10 * tmp10
    tmp32 = tmp11 - tmp31
    tmp33 = tmp30 * tmp32
    tmp34 = 0.7978845608028654
    tmp35 = tmp33 * tmp34
    tmp36 = 0.044715
    tmp37 = tmp35 * tmp36
    tmp38 = tmp7 * tmp7
    tmp39 = 3.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp37 * tmp40
    tmp42 = tmp35 + tmp41
    tmp43 = tmp29 * tmp12
    tmp44 = tmp43 * tmp8
    tmp45 = tmp42 + tmp44
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp45, rmask & xmask)
