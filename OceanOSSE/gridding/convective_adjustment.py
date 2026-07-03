import numpy as np
import gsw

def convective_adjustment(SA, CT, p, h):
    """
    Fully convectively mix an unstable water column.

    Parameters
    ----------
    SA : Absolute Salinity
    CT : Conservative Temperature
    p  : pressure (dbar)
    h  : layer thickness

    Returns
    -------
    SA_new, CT_new
    """
    SA = np.asarray(SA, dtype=np.float64)
    CT = np.asarray(CT, dtype=np.float64)
    p  = np.asarray(p,  dtype=np.float64)
    h  = np.asarray(h,  dtype=np.float64)

    SA_out = np.full_like(SA, np.nan)
    CT_out = np.full_like(CT, np.nan)

    # Find valid ocean points
    nwet = np.count_nonzero(np.isfinite(SA))

    SA = SA[:nwet]
    CT = CT[:nwet]
    p  = p[:nwet]
    h  = h[:nwet]
    
    assert SA.ndim == 1
    assert CT.ndim == 1
    assert p.ndim == 1
    assert h.ndim == 1

    # 1. FAST PATH: Check if the column is already stable in one go
    if nwet >= 2:
        rho_upper = gsw.density.rho(SA[:-1], CT[:-1], p[1:])
        rho_lower = gsw.density.rho(SA[1:], CT[1:], p[1:])
        if np.all(rho_upper <= rho_lower):
            SA_out[:nwet] = SA
            CT_out[:nwet] = CT
            return SA_out, CT_out

    # 2. SLOW PATH: Run stack-based merging algorithm (optimized)
    stack_top = []
    stack_bottom = []
    stack_h = []
    stack_heat = []
    stack_salt = []

    for k in range(nwet):
        stack_top.append(k)
        stack_bottom.append(k)
        stack_h.append(h[k])
        stack_heat.append(CT[k] * h[k])
        stack_salt.append(SA[k] * h[k])

        while len(stack_top) >= 2:
            top_lower = stack_top[-1]
            h_lower = stack_h[-1]
            heat_lower = stack_heat[-1]
            salt_lower = stack_salt[-1]

            h_upper = stack_h[-2]
            heat_upper = stack_heat[-2]
            salt_upper = stack_salt[-2]

            p_interface = p[top_lower]

            # Fast evaluation using a single call with 2 elements
            rhos = gsw.density.rho([salt_upper / h_upper, salt_lower / h_lower],
                                   [heat_upper / h_upper, heat_lower / h_lower],
                                   p_interface)
            rho_upper = rhos[0]
            rho_lower = rhos[1]

            if rho_upper <= rho_lower:
                break

            # Merge: upper gets the values
            stack_bottom[-2] = stack_bottom[-1]
            stack_h[-2] = h_upper + h_lower
            stack_heat[-2] = heat_upper + heat_lower
            stack_salt[-2] = salt_upper + salt_lower

            # Pop lower
            stack_top.pop()
            stack_bottom.pop()
            stack_h.pop()
            stack_heat.pop()
            stack_salt.pop()

    SA_new = np.empty_like(SA)
    CT_new = np.empty_like(CT)

    for i in range(len(stack_top)):
        top = stack_top[i]
        bottom = stack_bottom[i]
        SA_mix = stack_salt[i] / stack_h[i]
        CT_mix = stack_heat[i] / stack_h[i]

        SA_new[top:bottom+1] = SA_mix
        CT_new[top:bottom+1] = CT_mix

    SA_out[:nwet] = SA_new
    CT_out[:nwet] = CT_new

    return SA_out, CT_out

