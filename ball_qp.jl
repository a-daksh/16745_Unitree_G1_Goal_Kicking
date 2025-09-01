using LinearAlgebra
using OSQP
using Plots
using CairoMakie
using SparseArrays
function optimize_impulse(q_des, q_i; N=100, J_max=Inf, dt=0.01, m=0.43)
    g = 9.81  
    N -= 1 # 0 to 99 timesteps
    P = [(N/m*dt)^2 0;
         0 (N/m*dt)^2]
    q_T = (N*dt)/m .* [q_i[1] - q_des[1], q_i[2] - q_des[2] - (N-1)*N*dt^2*g/2]
    
    # Constraints
    G = [1.0 0.0;
         0.0 1.0;
         -1.0 0.0;
         0.0 -1.0]
    h = [J_max, J_max, 0.0, 0.0] # TODO tune max impulse
    
    # Create and solve QP problem
    model = OSQP.Model()
    OSQP.setup!(model; P=sparse(P), q=q_T, A=sparse(G), l=-Inf*ones(length(h)), u=h, verbose=false)
    res = OSQP.solve!(model)
    
    return res.x
end

function get_contact_point(J, r=0.11, ball_center=[0.0, 0.11])
    norm_J = J ./ norm(J)
    p_c = ball_center .- norm_J .* r
    return p_c
end

function visualize_contact(J, p_c; r=0.11, ball_center=[0.0, 0.11])
    norm_J = J ./ norm(J) .* r

     # Create figure and axis
    fig = Figure(size = (800, 600))  # Use size instead of resolution
    ax = Axis(fig[1, 1], limits=(ball_center[1] - 2 * r, ball_center[1] + 2 * r,ball_center[2] - 2 * r, ball_center[2] + 2 * r), title = "Ball Contact Point Visualization", xlabel = "X (m)", ylabel = "Z (m)")
    
    # Draw ball as a circle
    θ = LinRange(0, 2π, 100)
    ball_x = ball_center[1] .+ r * cos.(θ)
    ball_y = ball_center[2] .+ r * sin.(θ)
    lines!(ax, ball_x, ball_y, color=:blue, linewidth=2, label="Ball")
    
    # Plot elements (Contact Point, Ball Center, etc.)
    CairoMakie.scatter!(ax, [p_c[1]], [p_c[2]], color=:purple, marker=:x, label="Contact Point")
    CairoMakie.scatter!(ax, [ball_center[1]], [ball_center[2]], color=:red, label="Ball Center")
    
    # Impulse vector (norm_J)
    arrows!(ax, [p_c[1]], [p_c[2]], [norm_J[1]], [norm_J[2]], color=:green, label="Impulse Vector")
    
    fig[1, 2] = Legend(fig, ax, framevisible = false)

    
    # Show plot
    display(fig)
end

function dynamics_rollout(J, q_des, q_i; N=100, dt=0.01, m=0.43)
    v_i = J ./ m

    g = [0.0, 9.81]
    x = zeros(N, 4)
    
    x[1, :] = [q_i; v_i]
    for k in 2:N
        t = k-1  # Adjust for 1-based indexing
        v_k = v_i .- t*dt*g
        q_k = q_i .+ t*dt*J/m .- (t-1)*t*dt^2*g/2
        x[k, :] = [q_k; v_k]
    end
    @assert (norm(q_des .- x[end, 1:2], Inf)) < 1e-6
    # Create a plot using CairoMakie
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], title = "Dynamics Rollout", xlabel = "X (m)", ylabel = "Z (m)")

    # Plot position trajectory using lines
    lines!(ax, x[:, 1], x[:, 2], label="Position (q)")

    # Add the target point
    CairoMakie.scatter!(ax, [q_des[1]], [q_des[2]], color=:red, label="Target")

    # Display the plot
    display(fig)
end