using Printf
import ForwardDiff as FD

# Quick and dirty newton solver
function newton(residual, z_guess; active_inds = nothing, verbose = true, max_iters = 200, tol = 1e-8)
    if isnothing(active_inds)
        active_inds = 1:length(z_guess)
    end

    # Do Newton on the residual
    r = residual(z_guess)
    for iter = 1:max_iters
        dr_dz = FD.jacobian(residual, z_guess)[:, active_inds]

        Δz = -dr_dz \ r
        α = 1
        for i = 1:10
            z_candidate = copy(z_guess);
            z_candidate[active_inds] += α*Δz
            if norm(residual(z_candidate), 2) < norm(residual(z_guess), 2)
                break
            elseif i == 10
                @error "Linesearch failed"
                return z_guess
            end
            α = 0.5*α
        end

        z_guess[active_inds] += α*Δz
        r = residual(z_guess)

        if verbose
            @printf "iter = %d\tr = %1.2e\tcond(dr_dz) = %1.2e\tα = %1.2e\n" iter norm(r, Inf) cond(dr_dz) α
        end
        
        if norm(r, Inf) < tol
            break
        elseif iter == max_iters
            @warn @sprintf("Residual for iter = %d did not converge, ||r|| = %1.2e", iter, norm(r, Inf))
        end
    end
    return z_guess
end