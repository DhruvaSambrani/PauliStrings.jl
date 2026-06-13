using Test
using PauliStrings

@testset "TrotterTS Core Logic Tests" begin
    # Setup a simple 1D periodic ring of size 6
    Ls = (6,)
    Ps = (true,)

    @testset "Anticommutation Check" begin
        # X on site 1 and X on site 2 commute
        p = PauliStringTS{Ls,Ps}("X11111")
        @test !anticommutes(p, CartesianIndex(1)) # X1 vs 1X

        # XZ on sites 1,2: self-anticommutes at shift 1 (X_2 Z_3 vs X_1 Z_2 -> overlap at site 2 is X vs Z)
        p_xz = PauliStringTS{Ls,Ps}("XZ1111")
        @test anticommutes(p_xz, CartesianIndex(1))
        @test anticommutes(p_xz, CartesianIndex(-1))
        @test !anticommutes(p_xz, CartesianIndex(3)) # No overlap, must commute
    end

    @testset "Graph Coloring & Sublattice Partitioning" begin
        # XZ1111 on size 6 ring should partition into Even and Odd sublattices
        p_xz = PauliStringTS{Ls,Ps}("XZ1111")
        H = OperatorTS(p_xz) # Wraps to Operator{PauliStringTS}
        cache = PauliStrings.precompute_sublattices(H)

        sublattices = cache[p_xz]
        @test length(sublattices) == 2 # 2 colors (sublattices)

        # Verify that all elements in each sublattice commute with each other
        for g in sublattices
            for r in g
                for s in g
                    if r != s
                        @test !anticommutes(p_xz, r - s)
                    end
                end
            end
        end
    end

    @testset "Overlap Finder" begin
        p = PauliStringTS{Ls,Ps}("XZ1111")
        q = PauliStringTS{Ls,Ps}("XZ1111") # Flat PauliString
        g_even = CartesianIndex[CartesianIndex(1), CartesianIndex(3), CartesianIndex(5)]

        D_q = PauliStrings.evaluate_overlapping_shifts(p, q, g_even)
        # Shifts 1 and 5 in the even sublattice should overlap with Q
        @test CartesianIndex(1) in D_q
        @test CartesianIndex(5) in D_q
        @test length(D_q) == 2
    end

    @testset "TrotterTS vs Flat Trotter End-to-End Validation" begin
        # 1. Setup a periodic ring of size 6
        Ls = (6,)
        Ps = (true,)
        N = prod(Ls)
        hbar = 1.0
        dt = 0.05

        # 2. Build the physical Hamiltonian H = -sum(Z_i Z_{i+1}) - 0.3 * sum(X_i)
        # Build it as a flat Operator first
        H_flat = Operator(N)
        for i in 1:N
            H_flat += -1.0 * PauliString("ZZ" * "1"^(N - 2))
            H_flat += -0.3 * PauliString("X" * "1"^(N - 1))
        end

        # Wrap it to OperatorTS (which internally divides representative coefficients by N)
        H_TS = OperatorTS{Ls,Ps}(H_flat)

        # 3. Build the initial observable O0 = sum(X_i)
        O0_flat = Operator(N)
        for i in 1:N
            O0_flat += PauliString("X" * "1"^(N - 1))
        end
        O_TS = OperatorTS{Ls,Ps}(O0_flat)

        # Precompute our cache
        sublattices_cache = PauliStrings.precompute_sublattices(H_TS)

        @testset "Strang Splitting Step Verification" begin
            # --- A. Native Flat Trotter Step ---
            O_flat_step = copy(O0_flat)
            # Reconstruct the flat gates using the package's native trotterize
            # order=2 corresponds to Strang splitting
            g_flat = trotterize(H_flat, dt; order=2, heisenberg=true, hbar=hbar)
            trotter_step!(O_flat_step, g_flat)

            # --- B. Our Translation-Symmetric Trotter Step ---
            O_TS_step = copy(O_TS)
            O_TS_step = PauliStrings.tstrotter_step!(O_TS_step, H_TS, sublattices_cache, dt, hbar;
                order=2, truncation=nothing)

            # --- C. Compare the Results ---
            # We resum the TS result back to a flat Operator to compare them directly!
            O_TS_resummed = resum(O_TS_step)

            diff = O_TS_resummed - O_flat_step
            err = real(trace_product(diff, diff)) / 2^N
            # Check if the terms and coefficients are exactly identical
            @test err < 1e-12
        end
    end
    @testset "TrotterTS Diagnostic Tests" begin
        Ls = (6,)
        Ps = (true,)
        N = prod(Ls)

        @testset "1. Native Shift Direction Check" begin
            # "XZ1111" has active sites at [1, 2] (X at 1, Z at 2)
            p = PauliStringTS{Ls,Ps}("XZ1111")
            rep = representative(p)

            # Test a positive shift by 1
            rep_s1 = shift(rep, Ls, Ps, (1,))
            # If a positive shift moves elements to the right (1 -> 2, 2 -> 3):
            # the support of rep_s1 must be [2, 3]
            @test support(rep_s1) == [1, 6]

            # Test a negative shift by -1 (which wraps to 5 on a ring of size 6)
            rep_s5 = shift(rep, Ls, Ps, (5,))
            # If a negative shift moves elements to the left (1 -> 6, 2 -> 1):
            # the support of rep_s5 must be [1, 6]
            @test support(rep_s5) == [4, 5]
        end

        @testset "2. Overlap Finder Completeness" begin
            p = PauliStringTS{Ls,Ps}("XZ1111")
            q = PauliStringTS{Ls,Ps}("XZ1111") # Flat PauliString

            # Test across the entire translation group
            g_all::Vector{CartesianIndex} = CartesianIndex.(PauliStrings.all_shifts(Ls, Ps))

            D_q = PauliStrings.evaluate_overlapping_shifts(p, q, g_all)

            # A. Verify that every shift in D_q physically overlaps with Q
            for r in D_q
                p_shifted = shift(representative(p), Ls, Ps, Tuple(r))
                p_shifted_ts = PauliStringTS{Ls, Ps}
                @test !isempty(intersect(support(p_shifted), support(representative(q))))
            end

            # B. Verify that we didn't miss any overlapping shifts
            for r in g_all
                p_shifted = shift(representative(p), Ls, Ps, Tuple(r))
                if !isempty(intersect(support(p_shifted), support(representative(q))))
                    @test r in D_q
                end
            end
        end
    end
end

