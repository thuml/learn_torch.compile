
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_142', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6021120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3136) % 240
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x3 = (xindex // 752640)
    x6 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 57, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = x0
    tmp9 = tmp8 < tmp6
    tmp10 = tmp7 & tmp9
    tmp11 = tmp10 & tmp4
    tmp12 = tl.load(in_ptr0 + (x0 + (57*x1) + (3249*x2) + (194940*x3)), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp4, tmp14, tmp15)
    tmp17 = tmp0 >= tmp3
    tmp18 = tl.full([1], 120, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = 1 + x1
    tmp22 = tmp21 >= tmp1
    tmp23 = tl.full([1], 59, tl.int64)
    tmp24 = tmp21 < tmp23
    tmp25 = 1 + x0
    tmp26 = tmp25 >= tmp1
    tmp27 = tmp25 < tmp23
    tmp28 = tmp22 & tmp24
    tmp29 = tmp28 & tmp26
    tmp30 = tmp29 & tmp27
    tmp31 = tmp30 & tmp20
    tmp32 = tl.load(in_ptr1 + ((-208800) + x0 + (59*x1) + (3481*x2) + (208860*x3)), tmp31, other=0.0)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp31, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp20, tmp34, tmp35)
    tmp37 = tmp0 >= tmp18
    tmp38 = tl.full([1], 180, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = tmp37 & tmp39
    tmp41 = 2 + x1
    tmp42 = tmp41 >= tmp1
    tmp43 = tl.full([1], 61, tl.int64)
    tmp44 = tmp41 < tmp43
    tmp45 = 2 + x0
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp43
    tmp48 = tmp42 & tmp44
    tmp49 = tmp48 & tmp46
    tmp50 = tmp49 & tmp47
    tmp51 = tmp50 & tmp40
    tmp52 = tl.load(in_ptr2 + ((-446396) + x0 + (61*x1) + (3721*x2) + (223260*x3)), tmp51, other=0.0)
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp51, tmp52, tmp53)
    tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
    tmp56 = tl.where(tmp40, tmp54, tmp55)
    tmp57 = tmp0 >= tmp38
    tmp58 = tl.full([1], 240, tl.int64)
    tmp59 = tmp0 < tmp58
    tmp60 = 3 + x1
    tmp61 = tmp60 >= tmp1
    tmp62 = tl.full([1], 63, tl.int64)
    tmp63 = tmp60 < tmp62
    tmp64 = 3 + x0
    tmp65 = tmp64 >= tmp1
    tmp66 = tmp64 < tmp62
    tmp67 = tmp61 & tmp63
    tmp68 = tmp67 & tmp65
    tmp69 = tmp68 & tmp66
    tmp70 = tmp69 & tmp57
    tmp71 = tl.load(in_ptr3 + ((-714228) + x0 + (63*x1) + (3969*x2) + (238140*x3)), tmp70, other=0.0)
    tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
    tmp73 = tl.where(tmp70, tmp71, tmp72)
    tmp74 = tl.full(tmp73.shape, 0.0, tmp73.dtype)
    tmp75 = tl.where(tmp57, tmp73, tmp74)
    tmp76 = tl.where(tmp40, tmp56, tmp75)
    tmp77 = tl.where(tmp20, tmp36, tmp76)
    tmp78 = tl.where(tmp4, tmp16, tmp77)
    tl.store(out_ptr0 + (x6), tmp78, None)
