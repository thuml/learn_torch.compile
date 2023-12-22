
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_104', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768) % 8
    x2 = (xindex // 6144) % 8
    x3 = (xindex // 49152)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp1 = tl.load(in_ptr0 + (768 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp7 = tl.load(in_ptr0 + (1536 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp12 = tl.load(in_ptr0 + (13056 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp17 = tl.load(in_ptr0 + (13824 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp22 = tl.load(in_ptr0 + (14592 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp27 = tl.load(in_ptr0 + (26112 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp32 = tl.load(in_ptr0 + (26880 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp37 = tl.load(in_ptr0 + (27648 + x0 + (1536*x1) + (26112*x2) + (221952*x3)), None)
    tmp2 = tmp1 > tmp0
    tmp3 = 1 + (2*x1) + (34*x2)
    tmp4 = (2*x1) + (34*x2)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = 2 + (2*x1) + (34*x2)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = 17 + (2*x1) + (34*x2)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tmp18 = tmp17 > tmp16
    tmp19 = 18 + (2*x1) + (34*x2)
    tmp20 = tl.where(tmp18, tmp19, tmp15)
    tmp21 = triton_helpers.maximum(tmp17, tmp16)
    tmp23 = tmp22 > tmp21
    tmp24 = 19 + (2*x1) + (34*x2)
    tmp25 = tl.where(tmp23, tmp24, tmp20)
    tmp26 = triton_helpers.maximum(tmp22, tmp21)
    tmp28 = tmp27 > tmp26
    tmp29 = 34 + (2*x1) + (34*x2)
    tmp30 = tl.where(tmp28, tmp29, tmp25)
    tmp31 = triton_helpers.maximum(tmp27, tmp26)
    tmp33 = tmp32 > tmp31
    tmp34 = 35 + (2*x1) + (34*x2)
    tmp35 = tl.where(tmp33, tmp34, tmp30)
    tmp36 = triton_helpers.maximum(tmp32, tmp31)
    tmp38 = tmp37 > tmp36
    tmp39 = 36 + (2*x1) + (34*x2)
    tmp40 = tl.where(tmp38, tmp39, tmp35)
    tmp41 = triton_helpers.maximum(tmp37, tmp36)
    tl.store(out_ptr0 + (x4), tmp40, None)
