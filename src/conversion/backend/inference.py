from cnn_fixed_conversion_adapters import root_input_summary
from fixed_conversion_adapters import parallel_logits


def reset_inference_state(model,job):
    if job.is_dcgs:
        from DCGS.utils import reset
        reset(model)
        return "DCGS recursive neuron reset"
    if job.converter != "parallel":
        raise ValueError("Unsupported converter: " + job.converter)
    # ParaInfNeuron_CW_ND computes all slots without persistent membrane state.
    return "Parallel stateless temporal forward; no persistent membrane"


def forward_logits(model,x,job,counter=None,stop_time=None):
    root_input_summary(model,x,job)
    if job.is_dcgs:
        from DCGS.utils import reset
        reset(model);original_time=model.T
        if stop_time:
            # Execute only the requested prefix with the original64-calibrated neurons.
            # Encoder shape and per-neuron T are changed in a dedicated clone below.
            raise ValueError("DCGS prefix execution uses prefix_clone, not a runtime T override")
        out=model(x.clone())
        return [out[:i+1].mean(0) for i in range(job.time)]
    if job.converter=="parallel":
        if stop_time: raise ValueError("Parallel64 cannot be executed as a causal prefix")
        return parallel_logits(model,x,job.time)
    raise ValueError("Unsupported converter: " + job.converter)
