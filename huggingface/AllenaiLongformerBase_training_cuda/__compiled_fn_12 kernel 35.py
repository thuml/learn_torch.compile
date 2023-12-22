
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_select_backward_slice_backward_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9455616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 262656) % 3
    x1 = (xindex // 513) % 512
    x0 = xindex % 513
    x3 = (xindex // 787968)
    x4 = xindex % 262656
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x1
    tmp4 = tl.full([1], 255, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = x0
    tmp7 = tl.full([1], 258, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr0 + (256 + x4 + (525312*x3)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = 0.0
    tmp14 = tl.where(tmp8, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp5, tmp14, tmp15)
    tmp17 = tl.where(tmp5, tmp16, tmp13)
    tmp18 = tl.where(tmp2, tmp17, tmp13)
    tmp19 = tmp3 >= tmp4
    tmp20 = tl.full([1], 511, tl.int64)
    tmp21 = tmp3 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tl.full([1], 257, tl.int64)
    tmp24 = tmp6 >= tmp23
    tmp25 = tmp24 & tmp22
    tmp26 = tl.load(in_ptr1 + (256 + x4 + (131328*x2) + (525312*x3)), tmp25, other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp25, tmp26, tmp27)
    tmp29 = tl.where(tmp24, tmp28, tmp13)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp22, tmp29, tmp30)
    tmp32 = tl.where(tmp22, tmp31, tmp13)
    tmp33 = tmp18 + tmp32
    tmp34 = tl.full([1], 2, tl.int32)
    tmp35 = tmp0 == tmp34
    tmp36 = tl.full([1], 256, tl.int64)
    tmp37 = tmp3 >= tmp36
    tmp38 = tmp6 < tmp23
    tmp39 = tmp38 & tmp37
    tmp40 = tl.load(in_ptr2 + (262912 + x4 + (525312*x3)), tmp39, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp39, tmp40, tmp41)
    tmp43 = tl.where(tmp38, tmp42, tmp13)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp37, tmp43, tmp44)
    tmp46 = tl.where(tmp37, tmp45, tmp13)
    tmp47 = tl.where(tmp35, tmp46, tmp13)
    tmp48 = tmp33 + tmp47
    tl.store(out_ptr0 + (x5), tmp48, None)
