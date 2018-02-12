import PyMiniSolvers.minisolvers as minisolvers

def solve_sat(n_vars, iclauses):
    solver = minisolvers.MinisatSolver()
    for i in range(n_vars): solver.new_var(dvar=True)
    for iclause in iclauses: solver.add_clause(iclause)
    is_sat = solver.solve()
    stats = solver.get_stats()
    return is_sat, stats
