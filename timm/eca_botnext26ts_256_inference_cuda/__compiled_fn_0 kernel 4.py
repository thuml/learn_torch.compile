
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_silu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64) % 64
    x0 = xindex % 64
    x3 = (xindex // 64)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-129) + (2*x0) + (256*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, float("-inf"), tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = 2*x0
    tmp17 = tmp16 >= tmp1
    tmp18 = tmp16 < tmp3
    tmp19 = tmp17 & tmp18
    tmp20 = tmp5 & tmp19
    tmp21 = tl.load(in_ptr0 + ((-128) + (2*x0) + (256*x3)), tmp20, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.full(tmp23.shape, float("-inf"), tmp23.dtype)
    tmp25 = tl.where(tmp20, tmp23, tmp24)
    tmp26 = triton_helpers.maximum(tmp25, tmp15)
    tmp27 = 1 + (2*x0)
    tmp28 = tmp27 >= tmp1
    tmp29 = tmp27 < tmp3
    tmp30 = tmp28 & tmp29
    tmp31 = tmp5 & tmp30
    tmp32 = tl.load(in_ptr0 + ((-127) + (2*x0) + (256*x3)), tmp31, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.sigmoid(tmp32)
    tmp34 = tmp32 * tmp33
    tmp35 = tl.full(tmp34.shape, float("-inf"), tmp34.dtype)
    tmp36 = tl.where(tmp31, tmp34, tmp35)
    tmp37 = triton_helpers.maximum(tmp36, tmp26)
    tmp38 = 2*x1
    tmp39 = tmp38 >= tmp1
    tmp40 = tmp38 < tmp3
    tmp41 = tmp39 & tmp40
    tmp42 = tmp41 & tmp9
    tmp43 = tl.load(in_ptr0 + ((-1) + (2*x0) + (256*x3)), tmp42, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.sigmoid(tmp43)
    tmp45 = tmp43 * tmp44
    tmp46 = tl.full(tmp45.shape, float("-inf"), tmp45.dtype)
    tmp47 = tl.where(tmp42, tmp45, tmp46)
    tmp48 = triton_helpers.maximum(tmp47, tmp37)
    tmp49 = tmp41 & tmp19
    tmp50 = tl.load(in_ptr0 + ((2*x0) + (256*x3)), tmp49, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.sigmoid(tmp50)
    tmp52 = tmp50 * tmp51
    tmp53 = tl.full(tmp52.shape, float("-inf"), tmp52.dtype)
    tmp54 = tl.where(tmp49, tmp52, tmp53)
    tmp55 = triton_helpers.maximum(tmp54, tmp48)
    tmp56 = tmp41 & tmp30
    tmp57 = tl.load(in_ptr0 + (1 + (2*x0) + (256*x3)), tmp56, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.sigmoid(tmp57)
    tmp59 = tmp57 * tmp58
    tmp60 = tl.full(tmp59.shape, float("-inf"), tmp59.dtype)
    tmp61 = tl.where(tmp56, tmp59, tmp60)
    tmp62 = triton_helpers.maximum(tmp61, tmp55)
    tmp63 = 1 + (2*x1)
    tmp64 = tmp63 >= tmp1
    tmp65 = tmp63 < tmp3
    tmp66 = tmp64 & tmp65
    tmp67 = tmp66 & tmp9
    tmp68 = tl.load(in_ptr0 + (127 + (2*x0) + (256*x3)), tmp67, eviction_policy='evict_last', other=0.0)
    tmp69 = tl.sigmoid(tmp68)
    tmp70 = tmp68 * tmp69
    tmp71 = tl.full(tmp70.shape, float("-inf"), tmp70.dtype)
    tmp72 = tl.where(tmp67, tmp70, tmp71)
    tmp73 = triton_helpers.maximum(tmp72, tmp62)
    tmp74 = tmp66 & tmp19
    tmp75 = tl.load(in_ptr0 + (128 + (2*x0) + (256*x3)), tmp74, eviction_policy='evict_last', other=0.0)
    tmp76 = tl.sigmoid(tmp75)
    tmp77 = tmp75 * tmp76
    tmp78 = tl.full(tmp77.shape, float("-inf"), tmp77.dtype)
    tmp79 = tl.where(tmp74, tmp77, tmp78)
    tmp80 = triton_helpers.maximum(tmp79, tmp73)
    tmp81 = tmp66 & tmp30
    tmp82 = tl.load(in_ptr0 + (129 + (2*x0) + (256*x3)), tmp81, eviction_policy='evict_last', other=0.0)
    tmp83 = tl.sigmoid(tmp82)
    tmp84 = tmp82 * tmp83
    tmp85 = tl.full(tmp84.shape, float("-inf"), tmp84.dtype)
    tmp86 = tl.where(tmp81, tmp84, tmp85)
    tmp87 = triton_helpers.maximum(tmp86, tmp80)
    tl.store(out_ptr0 + (x4), tmp87, None)
