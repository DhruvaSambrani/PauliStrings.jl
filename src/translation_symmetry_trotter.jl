#=
function subtract_shift(r::CartesianIndex, delta::CartesianIndex, Ls::Tuple, Ps::Tuple)
    D = length(Ls)
    s = ntuple(D) do i
        if Ps[i]
            mod1(r[i] - delta[i], Ls[i])
        else
            val = r[i] - delta[i]
            if val < 1 || val > Ls[i]
                return -1
            end
            val
        end
    end
    any(==(-1), val) || return nothing
    return CartesianIndex(s)
end

function subtract_shift(x::T, delta::CartesianIndex, Ls::Tuple, Ps::Tuple) where T<:Integer
    W = sizeof(T) * 8
    shifted = x
    S = 1
    for d in eachindex(Ls)
        L = Ls[d]
        B = S * L
        shift_val = mod(delta[d], L)
        s = shift_val * S

        if s > 0
            if Ps[d]
                M_wrap = zero(T)
                base_mask = (one(T) << s) - one(T)
                for offset in 0:B:W-1
                    M_wrap |= (base_mask << offset)
                end
                shifted = ((shifted & M_wrap) << (B - s)) | ((shifted & ~M_wrap) >> s)
            else
                M_keep = zero(T)
                base_mask = ((one(T) << (B - s)) - one(T)) << s
                for offset in 0:B:W-1
                    M_keep |= (base_mask << offset)
                end
                shifted = (shifted & M_keep) >> s
            end
        end
        S *= L
    end
    return shifted
end

function anticommutes(p::PauliStringTS, delta::CartesianIndex)
    rep = representative(p)
    v_shifted = subtract_shift(rep.v, delta, p.Ls, p.Ps)
    w_shifted = subtract_shift(rep.w, delta, p.Ls, p.Ps)
    comm = (rep.v & w_shifted)^(rep.w & v_shifted)
    return isodd(count_ones(comm))
end

function find_self_anticommutations(p::PauliStringTS)
    qsz = qubitsize(p)
    grid = CartesianIndices(qsz)

    active = grid[support(representative(p))]

    anticommute_set = Set{eltype(grid)}()
    foreach(Iterators.product(active, active)) do (c1, c2)
        delta = c1 - c2
        if anticommutes(p, delta)
            push!(anticommute_set, delta)
        end
    end
    return anticommute_set
end

function precompute_sublattices(H::Operator{<:PauliStringTS})
    cache = Dict{PauliStringTS,Vector{Vector{CartesianIndex}}}()
    strings = H.strings

    for p in strings
        translations = all_shifts(p)
        anticommute_set = find_self_anticommutations(p)

        T_ci = eltype(translations)
        coloring = Dict{T_ci,Int}()
        sublattices = Vector{T_ci}[]

        for r in translations
            forbidden_colors = Set{Int}()

            for delta in anticommute_set
                s = subtract_shift(r, delta, p.Ls, p.Ps)

                if s !== nothing && haskey(coloring, s)
                    push!(forbidden_colors, coloring[s])
                end
            end

            color = 1
            while color in forbidden_colors
                color += 1
            end

            coloring[r] = color
            if color > length(sublattices)
                push!(sublattices, [r])
            else
                push!(sublattices[color], r)
            end
        end
        cache[p] = sublattices
    end
    return cache
end
=#

# TODO: should be a generic function
function anticommutes(p::PauliStringTS, delta::CartesianIndex)
    rep = representative(p)
    qsz = qubitsize(p)
    wrapped_shift = ntuple(i -> mod(delta[i], qsz[i]), length(qsz))
    rep_s = shift(representative(p), qsz, periodicflags(p), wrapped_shift)
    comm = (rep.v & rep_s.w) ⊻ (rep.w & rep_s.v)
    return isodd(count_ones(comm))
end

function precompute_sublattices(H::Operator{<:PauliStringTS})
    cache = Dict{PauliStringTS,Vector{Vector{CartesianIndex}}}()
    strings = H.strings

    for p in strings
        translations = CartesianIndex.(all_shifts(qubitsize(p), periodicflags(p)))
        T_ci = eltype(translations)

        coloring = Dict{T_ci,Int}()
        sublattices = Vector{T_ci}[]

        for r in translations
            forbidden_colors = Set{Int}()

            for (s, color_s) in coloring
                if anticommutes(p, r - s)
                    push!(forbidden_colors, color_s)
                end
            end

            color = 1
            while color in forbidden_colors
                color += 1
            end

            coloring[r] = color
            if color > length(sublattices)
                push!(sublattices, [r])
            else
                push!(sublattices[color], r)
            end
        end
        cache[p] = sublattices
    end
    return cache
end

function tstrotter_step!(O::Operator{<:PauliStringTS}, H::Operator{<:PauliStringTS}, sublattices_cache, dt, hbar;
    order, truncation)
    coeffs = H.coeffs
    strings = H.strings
    m = length(strings)

    if order == 1
        for a in 1:m
            p = strings[a]
            c = coeffs[a]
            sublattices = sublattices_cache[p]
            for g in sublattices
                O = apply_sublattice_step(O, p, g, real(c), dt, hbar; truncation=truncation)
            end
        end
    elseif order == 2
        # Symmetric Strang splitting (Forward pass)
        for a in 1:m
            p = strings[a]
            c = coeffs[a]
            sublattices = sublattices_cache[p]
            for g in sublattices
                O = apply_sublattice_step(O, p, g, real(c), dt / 2, hbar; truncation=truncation)
            end
        end
        # Symmetric Strang splitting (Backward pass in reverse order)
        for a in m:-1:1
            p = strings[a]
            c = coeffs[a]
            sublattices = sublattices_cache[p]
            for g in reverse(sublattices)
                O = apply_sublattice_step(O, p, g, real(c), dt / 2, hbar; truncation=truncation)
            end
        end
    else
        error("Only order 1 and 2 Trotter are supported.")
    end

    return O
end

function apply_sublattice_step(O::OperatorTS{Ls,Ts}, p::PauliStringTS, g::Vector{CartesianIndex}, c::Real, dt::Real, hbar::Real; truncation) where {Ls,Ts}
    N = Base.prod(Ls)
    θ = 2 * c * dt * N / hbar
    O_new_flat = Operator(qubitlength(O))
    for (o_val, q) in zip(O.coeffs, O.strings)
        D_q = evaluate_overlapping_shifts(p, q, g)
        current_op = Operator(representative(q))
        for delta in D_q
            qsz_p = qubitsize(p)
            wrapped_delta = ntuple(i -> mod(delta[i], qsz_p[i]), length(qsz_p))
            P_delta = shift(representative(p), qsz_p, periodicflags(p), wrapped_delta)
            current_op = mapreduce(+, zip(current_op.coeffs, current_op.strings)) do (coeff, s)
                coeff * pauli_rotation(P_delta, s, θ)
            end
        end
        O_new_flat += o_val * current_op
    end
    O_new = OperatorTS{Ls,Ts}(O_new_flat)
    return O_new
end
function evaluate_overlapping_shifts(p::PauliStringTS, q::PauliStringTS, g::Vector{CartesianIndex})
    qsp = qubitsize(p)
    pfp = periodicflags(p)
    S_p = CartesianIndices(qsp)[support(representative(p))]
    S_q = CartesianIndices(qubitsize(q))[support(representative(q))]

    D_q = Set{CartesianIndex}()
    for x in S_p
        for y in S_q
            # Compute the coordinate shift r that maps x to y (wrapped under periodic boundaries)
            r = ntuple(length(qsp)) do i
                if pfp[i]
                    mod1(y[i] - x[i], qsp[i])
                else
                    y[i] - x[i]
                end
            end
            r_ci = CartesianIndex(r)
            if r_ci in g
                push!(D_q, r_ci)
            end
        end
    end
    return D_q
end
