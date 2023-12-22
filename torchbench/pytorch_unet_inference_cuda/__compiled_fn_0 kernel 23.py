
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[134217728], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 78479360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 958) % 640
    x0 = xindex % 958
    x2 = (xindex // 613120)
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.49921752738654146
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.int32)
    tmp9 = x0
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = 0.4994775339602926
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14.to(tl.int32)
    tmp16 = tl.load(in_ptr0 + (tmp15 + (479*tmp8) + (153280*x2)), None, eviction_policy='evict_last')
    tmp17 = tmp8.to(tl.float32)
    tmp18 = tmp7 - tmp17
    tmp19 = tmp2 - tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = tl.math.ceil(tmp7)
    tmp22 = 319.0
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tmp24 = tmp23.to(tl.int32)
    tmp25 = tl.load(in_ptr0 + (tmp15 + (479*tmp24) + (153280*x2)), None, eviction_policy='evict_last')
    tmp26 = tmp25 * tmp18
    tmp27 = tmp20 + tmp26
    tmp28 = tl.math.ceil(tmp14)
    tmp29 = 478.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = tmp30.to(tl.int32)
    tmp32 = tl.load(in_ptr0 + (tmp31 + (479*tmp8) + (153280*x2)), None, eviction_policy='evict_last')
    tmp33 = tmp32 * tmp19
    tmp34 = tl.load(in_ptr0 + (tmp31 + (479*tmp24) + (153280*x2)), None, eviction_policy='evict_last')
    tmp35 = tmp34 * tmp18
    tmp36 = tmp15.to(tl.float32)
    tmp37 = tmp14 - tmp36
    tmp38 = tmp2 - tmp37
    tmp39 = tmp27 * tmp38
    tmp40 = tmp33 + tmp35
    tmp41 = tmp40 * tmp37
    tmp42 = tmp39 + tmp41
    tl.store(in_out_ptr0 + (x4), tmp42, None)
