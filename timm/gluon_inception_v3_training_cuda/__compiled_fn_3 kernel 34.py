
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2728448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 73
    x2 = (xindex // 4672) % 73
    x3 = (xindex // 341056)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x1) + (18816*x2) + (1382976*x3)), xmask)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), xmask)
    tmp3 = tl.load(in_ptr0 + (128 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), xmask)
    tmp5 = tl.load(in_ptr0 + (9408 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), xmask)
    tmp7 = tl.load(in_ptr0 + (9472 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), xmask)
    tmp9 = tl.load(in_ptr0 + (9536 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), xmask)
    tmp11 = tl.load(in_ptr0 + (18816 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), xmask)
    tmp13 = tl.load(in_ptr0 + (18880 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), xmask)
    tmp15 = tl.load(in_ptr0 + (18944 + x0 + (128*x1) + (18816*x2) + (1382976*x3)), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = 1 + (2*x1) + (294*x2)
    tmp19 = (2*x1) + (294*x2)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = 2 + (2*x1) + (294*x2)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = 147 + (2*x1) + (294*x2)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = 148 + (2*x1) + (294*x2)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = 149 + (2*x1) + (294*x2)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = 294 + (2*x1) + (294*x2)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = 295 + (2*x1) + (294*x2)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = 296 + (2*x1) + (294*x2)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x4), tmp16, xmask)
    tl.store(out_ptr1 + (x4), tmp41, xmask)
