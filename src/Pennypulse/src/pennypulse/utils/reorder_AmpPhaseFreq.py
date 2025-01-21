from pennylane.pulse.transmon import AmplitudeAndPhaseAndFreq


def _reorder_AmpPhaseFreq(params, coeffs_parametrized):
    """Takes `params`, and reorganizes it based on whether the Hamiltonian has
    callable phase and/or callable amplitude and/or callable freq.

    Consolidates amplitude, phase and freq parameters if they are callable,
    and duplicates parameters since they will be passed to two operators in the Hamiltonian"""

    if len(params) == 0:
        return params

    reordered_params = []

    coeff_idx = 0
    params_idx = 0

    if len(coeffs_parametrized) == 0:
        return reordered_params

    for i, coeff in enumerate(coeffs_parametrized):
        if i == coeff_idx:
            if isinstance(coeff, AmplitudeAndPhaseAndFreq):
                is_callables = [
                    coeff.phase_is_callable,
                    coeff.amp_is_callable,
                    coeff.freq_is_callable,
                ]

                num_callables = sum(is_callables)

                # package parameters according to how many coeffs are callable
                reordered_params.extend([params[params_idx: params_idx + num_callables]])

                coeff_idx += 1
                params_idx += num_callables

            else:
                reordered_params.append(params[params_idx])
                coeff_idx += 1
                params_idx += 1

    return reordered_params
