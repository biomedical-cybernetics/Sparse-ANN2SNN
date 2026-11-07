import copy
import torch
from types import SimpleNamespace
from fixed_conversion_adapters import fold_bn
worker=SimpleNamespace(fold_bn=fold_bn)

def fusion_audit(ann,folded,x,before,after):
    result={"primary_allclose":torch.allclose(before,after,rtol=1e-4,atol=1e-4),
        "fp32_max_abs_difference":float((before-after).abs().max()),"fallback_used":False}
    if result["primary_allclose"]:return result
    reference=copy.deepcopy(ann).double().eval()
    reference_folded=worker.fold_bn(copy.deepcopy(reference)).eval()
    unfused64=reference(x.double());fused64=reference_folded(x.double())
    bound=2e-5*max(1.,float(unfused64.abs().max()))
    source_error=float((before.double()-unfused64).abs().max());fused_error=float((after.double()-unfused64).abs().max())
    result.update(fallback_used=True,reference_precision="FP64 audit copy only",fp32_bound=bound,
        fp64_formula_max_abs_difference=float((unfused64-fused64).abs().max()),
        fp64_formula_allclose=torch.allclose(unfused64,fused64,rtol=1e-9,atol=1e-9),
        fp32_source_reference_error=source_error,fp32_fused_reference_error=fused_error,
        prediction_mismatches=int((before.argmax(1)!=after.argmax(1)).sum()),
        source_reference_prediction_mismatches=int((before.argmax(1)!=unfused64.argmax(1)).sum()),
        samples=len(x),operational_precision="FP32",weights_or_fusion_formula_changed=False)
    if (not result["fp64_formula_allclose"] or max(source_error,fused_error,result["fp32_max_abs_difference"])>bound
            or result["prediction_mismatches"] or result["source_reference_prediction_mismatches"]):
        raise ValueError("BN fusion validation failed: "+str(result))
    return result
