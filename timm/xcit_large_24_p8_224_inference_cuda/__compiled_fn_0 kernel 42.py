
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (602880*x1)), None)
    tmp20 = tl.load(in_ptr4 + (785*x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (785*x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp1 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp1 < tmp3
    tmp5 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr2 + (x2), tmp4, other=0.0)
    tmp7 = tl.load(in_ptr3 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp1 >= tmp3
    tmp13 = tl.full([1], 785, tl.int64)
    tmp14 = tmp1 < tmp13
    tmp15 = tl.load(in_ptr0 + (x0 + (602880*x1)), tmp12, other=0.0)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp12, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp11, tmp17)
    tmp19 = tmp0 + tmp18
    tmp21 = tmp19 - tmp20
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = 1e-06
    tmp26 = tmp24 + tmp25
    tmp27 = tl.math.rsqrt(tmp26)
    tmp28 = tmp21 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tl.store(out_ptr0 + (x2), tmp32, None)
