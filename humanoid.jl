# Modified version of 16-675 HW1 quadruped.jl
using RigidBodyDynamics
using MeshCat
using MeshCatMechanisms
using Random
using StaticArrays
using Rotations
using LinearAlgebra
using ForwardDiff
using Quaternions
using CoordinateTransformations

const URDFPATH = joinpath(@__DIR__, "unitree_robots", "g1", "urdf", "g1_29dof.urdf")

function attach_left_ankle!(mech::Mechanism{T}; revolute::Bool=true, fixed::Bool=false) where T
    # Get the relevant bodies and joints
    left_ankle_body = findbody(mech, "left_ankle_roll_link")  # Use the correct body name
    world_body = findbody(mech, "world")
    trunk = findbody(mech, "pelvis")  # The world frame is the root body
    state = MechanismState(mech)

    trunk_to_ankle = RigidBodyDynamics.translation(relative_transform(state, default_frame(trunk), default_frame(left_ankle_body)))
    #body_z_offset = 0 #-0.01755
    foot_bottom_offset = -0.03
    left_ankle_location = SA[trunk_to_ankle[1], trunk_to_ankle[2], foot_bottom_offset]
    
    if !revolute && !fixed
        println("here")
        # Create a spherical joint to allow free rotation
        ankle_joint = Joint("left_ankle_to_world", QuaternionSpherical{T}())
        joint_pose = Transform3D(
            frame_before(ankle_joint),
            default_frame(world_body),
            -left_ankle_location  # Position at origin of world frame
        )

        # Attach the left ankle body to the world using this joint
        attach!(mech, world_body, left_ankle_body, ankle_joint, joint_pose=joint_pose)
    elseif fixed
        ankle_joint = Joint("left_ankle_to_world", RigidBodyDynamics.Fixed{T}())
        joint_pose = Transform3D(
            frame_before(ankle_joint),
            default_frame(world_body),
            -left_ankle_location
        )
        attach!(mech, world_body, left_ankle_body, ankle_joint, joint_pose=joint_pose)
    else
        # Create revolute joints for X, Y, and Z axes
        dummy1 = RigidBody{T}("dummy1")
        dummy2 = RigidBody{T}("dummy2")

        # Assign spatial inertia to dummy bodies (minimal mass and moment)
        for body in (dummy1, dummy2)
            inertia = SpatialInertia(default_frame(body),
                moment=I(3) * 1e-3,
                com=SVector(0., 0., 0.),
                mass=1e-3
            )
            spatial_inertia!(body, inertia)
        end

        # X-axis revolute joint
        ankle_joint_x = Joint("left_ankle_joint_x", Revolute{T}(SVector(1., 0., 0.)))
        joint_pose_x = Transform3D(
            frame_before(ankle_joint_x),
            default_frame(world_body),
            -left_ankle_location
        )
        attach!(mech, world_body, dummy1, ankle_joint_x, joint_pose=joint_pose_x)

        # Y-axis revolute joint
        ankle_joint_y = Joint("left_ankle_joint_y", Revolute{T}(SVector(0., 1., 0.)))
        joint_pose_y = Transform3D(
            frame_before(ankle_joint_y),
            default_frame(dummy1),
            SVector(0., 0., 0.)
        )
        attach!(mech, dummy1, dummy2, ankle_joint_y, joint_pose=joint_pose_y)

        # Z-axis revolute joint
        ankle_joint_z = Joint("left_ankle_joint_z", Revolute{T}(SVector(0., 0., 1.)))
        joint_pose_z = Transform3D(
            frame_before(ankle_joint_z),
            default_frame(dummy2),
            SVector(0., 0., 0.)
        )
        attach!(mech, dummy2, left_ankle_body, ankle_joint_z, joint_pose=joint_pose_z)
    end

    # Remove the floating base joint if it exists, TODO: maybe modify this if needed
    floating_base_joint = findjoint(mech, "pelvis_to_world")
    if floating_base_joint !== nothing
        remove_joint!(mech, floating_base_joint)
    end
end

function build_humanoid()
    g1 = parse_urdf(URDFPATH, floating=true, remove_fixed_tree_joints=false)
    attach_left_ankle!(g1, revolute=true, fixed=true)
    return g1
end

struct G1Humanoid{C}
    mech::Mechanism{Float64}
    statecache::C
    dyncache::DynamicsResultCache{Float64}
    xdot::Vector{Float64}
    function G1Humanoid(mech::Mechanism)
        N = num_positions(mech) + num_velocities(mech)
        statecache = StateCache(mech)
        rescache = DynamicsResultCache(mech)
        xdot = zeros(N)
        new{typeof(statecache)}(mech, statecache, rescache, xdot)
    end
end

function G1Humanoid()
    G1Humanoid(build_humanoid())
end

state_dim(model::G1Humanoid) = 58  # 29 joints * 2 (position and velocity)
control_dim(model::G1Humanoid) = 29  # 29 actuated joints

function get_partition(model::G1Humanoid)
    n,m = state_dim(model), control_dim(model)
    return 1:n, n .+ (1:m), n+m .+ (1:n)
end

function dynamics(model::G1Humanoid, x::AbstractVector{T1}, u::AbstractVector{T2}) where {T1,T2}
    T = promote_type(T1,T2)
    state = model.statecache[T]
    res = model.dyncache[T]

    copyto!(state, x)
    τ = zeros(T, num_velocities(model.mech))
    
    # Assuming the first 6 DoFs are for the floating base (unactuated)
    # and the rest are actuated joints
    τ = 1 * u    
    dynamics!(res, state, τ)
    q̇ = res.q̇
    v̇ = res.v̇
    return [q̇; v̇]
end

function jacobian(model::G1Humanoid, x, u)
    ix = SVector{64}(1:64)
    iu = SVector{29}(65:93)
    faug(z) = dynamics(model, z[ix], z[iu])
    z = [x; u]
    ForwardDiff.jacobian(faug, z)
end

# Set initial guess
function initial_state(model::G1Humanoid)
    state = model.statecache[Float64]
    g1 = model.mech
    zero!(state)  # Initialize state to zero

    function set_joint_configuration!(state, g1, joint_name, value)
        joint = findjoint(g1, joint_name)
        if joint !== nothing
            if isa(joint, Joint{Float64, QuaternionFloating{Float64}})
                angle = deg2rad(value)
                axis = SVector{3, Float64}(1.0, 0.0, 0.0)  # Rotation around X-axis of joint
                aa = AngleAxis(angle, axis...)
                quat = QuatRotation(aa)
                rot_matrix = RotMatrix{3, Float64}(quat)
                transform = Transform3D(frame_before(joint), frame_after(joint), rot_matrix, SVector{3, Float64}(0.0, 0.0, 0.0))
                set_configuration!(state, joint, transform)
            else
                set_configuration!(state, joint, deg2rad(value))
            end
        else
            println("Joint not found: $joint_name")
        end
    end
    # Set configurations for each joint
    #set_joint_configuration!(state, g1, "pelvis_to_world", 0)
    #set_joint_configuration!(state, g1, "pelvis_contour_joint", 0)

    # Hip Joints
    set_joint_configuration!(state, g1, "left_hip_pitch_joint", -40)
    set_joint_configuration!(state, g1, "right_hip_pitch_joint", -20)
    set_joint_configuration!(state, g1, "left_hip_roll_joint", 20)
    set_joint_configuration!(state, g1, "right_hip_roll_joint", 0)
    set_joint_configuration!(state, g1, "left_hip_yaw_joint", 0)
    set_joint_configuration!(state, g1, "right_hip_yaw_joint", 0)

    # Waist Joints
    set_joint_configuration!(state, g1, "waist_yaw_joint", 0)
    set_joint_configuration!(state, g1, "waist_roll_joint", 0)
    set_joint_configuration!(state, g1, "waist_pitch_joint", 0)

    # Knee Joints
    set_joint_configuration!(state, g1, "left_knee_joint", 30)
    set_joint_configuration!(state, g1, "right_knee_joint", 80)

    # Ankle Joints
    set_joint_configuration!(state, g1, "left_ankle_pitch_joint", -10)
    set_joint_configuration!(state, g1, "right_ankle_pitch_joint", 10)
    set_joint_configuration!(state, g1, "left_ankle_roll_joint", 0)
    set_joint_configuration!(state, g1, "right_ankle_roll_joint", 0)

    # Shoulder Joints
    set_joint_configuration!(state, g1, "left_shoulder_pitch_joint", 10)
    set_joint_configuration!(state, g1, "right_shoulder_pitch_joint", 10)
    set_joint_configuration!(state, g1, "left_shoulder_roll_joint", 5)
    set_joint_configuration!(state, g1, "right_shoulder_roll_joint", 5)
    set_joint_configuration!(state, g1, "left_shoulder_yaw_joint", 5)
    set_joint_configuration!(state, g1, "right_shoulder_yaw_joint", 5)

    # Elbow Joints
    set_joint_configuration!(state, g1, "left_elbow_joint", -30)
    set_joint_configuration!(state, g1, "right_elbow_joint", -30)

    # Wrist Joints
    set_joint_configuration!(state, g1, "left_wrist_roll_joint", 10)
    set_joint_configuration!(state, g1, "right_wrist_roll_joint", 10)
    set_joint_configuration!(state, g1, "left_wrist_pitch_joint", 0)
    set_joint_configuration!(state, g1, "right_wrist_pitch_joint", 0)
    set_joint_configuration!(state, g1, "left_wrist_yaw_joint", 0)
    set_joint_configuration!(state, g1, "right_wrist_yaw_joint", 0)

    # Hand Joints
    #set_joint_configuration!(state, g1, "left_hand_palm_joint", 0)
    #set_joint_configuration!(state, g1, "right_hand_palm_joint", 0)

    # Other Joints
    #set_joint_configuration!(state, g1, "logo_joint", 0)
    #set_joint_configuration!(state, g1, "head_joint", 0)
    #set_joint_configuration!(state, g1, "imu_in_pelvis_joint", 0)
    #set_joint_configuration!(state, g1, "imu_in_torso_joint", 0)
    #set_joint_configuration!(state, g1, "d435_joint", 0)
    #set_joint_configuration!(state, g1, "mid360_joint", 0)
    #set_joint_configuration!(state, g1, "waist_support_joint", 0)
    # set_configuration!(state, findjoint(g1, "left_ankle_joint_x"), deg2rad(00))
    # set_configuration!(state, findjoint(g1, "left_ankle_joint_y"), deg2rad(-00))
    # set_configuration!(state, findjoint(g1, "left_ankle_joint_z"), deg2rad(-00))

    return [configuration(state); velocity(state)]
end

function initialize_visualizer(g1::G1Humanoid)
    vis = Visualizer()
    delete!(vis)
    cd(joinpath(@__DIR__, "unitree_robots","g1","urdf"))
    mvis = MechanismVisualizer(g1.mech, URDFVisuals(URDFPATH), vis)
    cd(@__DIR__)
    return mvis
end


function get_right_foot_tip_location(mech::Mechanism{T1}, x::AbstractVector{T2}) where {T1, T2<:Real}
    state = MechanismState{T2}(mech)
    copyto!(state, x)
    
    # Get the right ankle roll link body
    right_ankle_body = findbody(mech, "right_ankle_roll_link")
    world_body = findbody(mech, "world")

    # Get the transformation from the world frame to the right ankle body
    right_ankle_to_world = relative_transform(state, default_frame(right_ankle_body), default_frame(world_body))
    foot_tip_offset = Transform3D(
        default_frame(right_ankle_body), 
        default_frame(right_ankle_body),
        RotMatrix3(one(SMatrix{3,3,T2})),
        SVector{3,T2}(0.14, 0.0, -0.03)
    )
    foot_tip_in_world = right_ankle_to_world * foot_tip_offset
    return RigidBodyDynamics.translation(foot_tip_in_world)
end