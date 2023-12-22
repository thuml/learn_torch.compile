
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_86', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128)
    x0 = xindex % 128
    x2 = xindex
    tmp14 = tl.load(in_ptr1 + (x2), None)
    tmp16 = tl.load(in_ptr2 + (x2), None)
    tmp17 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp0 = ((x1 % 196) // 14) % 2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = ((x1 % 196) % 14) % 2
    tmp4 = tmp3 == tmp1
    tmp5 = tmp4 & tmp2
    tmp6 = tl.load(in_ptr0 + (x0 + (128*(((x1 % 196) % 14) // 2)) + (896*((x1 % 196) // 28)) + (6272*(x1 // 196))), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 0.0
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = tl.where(tmp2, tmp12, tmp9)
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.0006377551020408163
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.store(out_ptr0 + (x2), tmp32, None)
