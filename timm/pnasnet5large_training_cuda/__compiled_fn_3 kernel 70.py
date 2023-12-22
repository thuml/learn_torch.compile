
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_relu_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3048192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x3 = (xindex // 9072) % 42
    x2 = (xindex // 216) % 42
    x7 = (xindex // 216) % 1764
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = (-1) + x3
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp2 >= tmp3
    tmp5 = tl.full([1], 42, tl.int64)
    tmp6 = tmp2 < tmp5
    tmp7 = tmp4 & tmp6
    tmp8 = (-1) + x2
    tmp9 = tmp8 >= tmp3
    tmp10 = tmp8 < tmp5
    tmp11 = tmp9 & tmp10
    tmp12 = tmp7 & tmp11
    tmp13 = tl.load(in_ptr0 + ((-9288) + x0), tmp12 & xmask, other=0.0)
    tmp14 = tl.full(tmp13.shape, float("-inf"), tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = x2
    tmp17 = tmp16 >= tmp3
    tmp18 = tmp16 < tmp5
    tmp19 = tmp17 & tmp18
    tmp20 = tmp7 & tmp19
    tmp21 = tl.load(in_ptr0 + ((-9072) + x0), tmp20 & xmask, other=0.0)
    tmp22 = tl.full(tmp21.shape, float("-inf"), tmp21.dtype)
    tmp23 = tl.where(tmp20, tmp21, tmp22)
    tmp24 = triton_helpers.maximum(tmp23, tmp15)
    tmp25 = 1 + x2
    tmp26 = tmp25 >= tmp3
    tmp27 = tmp25 < tmp5
    tmp28 = tmp26 & tmp27
    tmp29 = tmp7 & tmp28
    tmp30 = tl.load(in_ptr0 + ((-8856) + x0), tmp29 & xmask, other=0.0)
    tmp31 = tl.full(tmp30.shape, float("-inf"), tmp30.dtype)
    tmp32 = tl.where(tmp29, tmp30, tmp31)
    tmp33 = triton_helpers.maximum(tmp32, tmp24)
    tmp34 = x3
    tmp35 = tmp34 >= tmp3
    tmp36 = tmp34 < tmp5
    tmp37 = tmp35 & tmp36
    tmp38 = tmp37 & tmp11
    tmp39 = tl.load(in_ptr0 + ((-216) + x0), tmp38 & xmask, other=0.0)
    tmp40 = tl.full(tmp39.shape, float("-inf"), tmp39.dtype)
    tmp41 = tl.where(tmp38, tmp39, tmp40)
    tmp42 = triton_helpers.maximum(tmp41, tmp33)
    tmp43 = tmp37 & tmp19
    tmp44 = tl.load(in_ptr0 + (x0), tmp43 & xmask, other=0.0)
    tmp45 = tl.full(tmp44.shape, float("-inf"), tmp44.dtype)
    tmp46 = tl.where(tmp43, tmp44, tmp45)
    tmp47 = triton_helpers.maximum(tmp46, tmp42)
    tmp48 = tmp37 & tmp28
    tmp49 = tl.load(in_ptr0 + (216 + x0), tmp48 & xmask, other=0.0)
    tmp50 = tl.full(tmp49.shape, float("-inf"), tmp49.dtype)
    tmp51 = tl.where(tmp48, tmp49, tmp50)
    tmp52 = triton_helpers.maximum(tmp51, tmp47)
    tmp53 = 1 + x3
    tmp54 = tmp53 >= tmp3
    tmp55 = tmp53 < tmp5
    tmp56 = tmp54 & tmp55
    tmp57 = tmp56 & tmp11
    tmp58 = tl.load(in_ptr0 + (8856 + x0), tmp57 & xmask, other=0.0)
    tmp59 = tl.full(tmp58.shape, float("-inf"), tmp58.dtype)
    tmp60 = tl.where(tmp57, tmp58, tmp59)
    tmp61 = triton_helpers.maximum(tmp60, tmp52)
    tmp62 = tmp56 & tmp19
    tmp63 = tl.load(in_ptr0 + (9072 + x0), tmp62 & xmask, other=0.0)
    tmp64 = tl.full(tmp63.shape, float("-inf"), tmp63.dtype)
    tmp65 = tl.where(tmp62, tmp63, tmp64)
    tmp66 = triton_helpers.maximum(tmp65, tmp61)
    tmp67 = tmp56 & tmp28
    tmp68 = tl.load(in_ptr0 + (9288 + x0), tmp67 & xmask, other=0.0)
    tmp69 = tl.full(tmp68.shape, float("-inf"), tmp68.dtype)
    tmp70 = tl.where(tmp67, tmp68, tmp69)
    tmp71 = triton_helpers.maximum(tmp70, tmp66)
    tmp72 = tmp23 > tmp15
    tmp73 = (-42) + x7
    tmp74 = (-43) + x7
    tmp75 = tl.where(tmp72, tmp73, tmp74)
    tmp76 = tmp32 > tmp24
    tmp77 = (-41) + x7
    tmp78 = tl.where(tmp76, tmp77, tmp75)
    tmp79 = tmp41 > tmp33
    tmp80 = (-1) + x7
    tmp81 = tl.where(tmp79, tmp80, tmp78)
    tmp82 = tmp46 > tmp42
    tmp83 = x7
    tmp84 = tl.where(tmp82, tmp83, tmp81)
    tmp85 = tmp51 > tmp47
    tmp86 = 1 + x7
    tmp87 = tl.where(tmp85, tmp86, tmp84)
    tmp88 = tmp60 > tmp52
    tmp89 = 41 + x7
    tmp90 = tl.where(tmp88, tmp89, tmp87)
    tmp91 = tmp65 > tmp61
    tmp92 = 42 + x7
    tmp93 = tl.where(tmp91, tmp92, tmp90)
    tmp94 = tmp70 > tmp66
    tmp95 = 43 + x7
    tmp96 = tl.where(tmp94, tmp95, tmp93)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
    tl.store(out_ptr1 + (x0), tmp71, xmask)
    tl.store(out_ptr2 + (x0), tmp96, xmask)
