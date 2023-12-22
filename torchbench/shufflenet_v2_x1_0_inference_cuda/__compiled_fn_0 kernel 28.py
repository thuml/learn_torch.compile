
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 90944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = (xindex // 49) % 464
    x5 = xindex
    x3 = (xindex // 22736)
    x6 = xindex % 22736
    x0 = xindex % 49
    x1 = (xindex // 49) % 232
    x2 = (xindex // 11368) % 2
    tmp0 = x4
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 232, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x5), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 464, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-11368) + x6 + (11368*x3)), tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-232) + x4), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + ((-232) + x4), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + ((-232) + x4), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + ((-232) + x4), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = triton_helpers.maximum(0, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tl.store(out_ptr0 + (x0 + (49*x2) + (98*x1) + (22736*x3)), tmp29, xmask)
