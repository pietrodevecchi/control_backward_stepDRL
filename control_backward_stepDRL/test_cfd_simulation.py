from cfd_solver import FlowSolver, profile
from dolfin import *
import time

import csv



# list of solvers

# bicgstab       |  Biconjugate gradient stabilized method      
# cg             |  Conjugate gradient method                   
# default        |  default Krylov method                       
# gmres          |  Generalized minimal residual method         
# minres         |  Minimal residual method                     
# richardson     |  Richardson method                           
# tfqmr          |  Transpose-free quasi-minimal residual method

# list of preconditioners

# amg              |  Algebraic multigrid                       
# default          |  default preconditioner                    
# hypre_amg        |  Hypre algebraic multigrid (BoomerAMG)     
# hypre_euclid     |  Hypre parallel incomplete LU factorization
# hypre_parasails  |  Hypre parallel sparse approximate inverse 
# icc              |  Incomplete Cholesky factorization         
# ilu              |  Incomplete LU factorization               
# jacobi           |  Jacobi iteration                          
# none             |  No preconditioner                         
# petsc_amg        |  PETSc algebraic multigrid                 
# sor              |  Successive over-relaxation     

# 7 solvers
list_of_solvers = ['bicgstab', 'cg', 'default', 'gmres', 'minres', 'richardson', 'tfqmr']

# 11 preconditioners
list_of_preconditioners = ['amg', 'default', 'hypre_amg', 'hypre_euclid',
                           'hypre_parasails', 'icc', 'ilu', 'jacobi', 'none', 'persc_amg', 'sor']


geometry_params = {'frequency': 10,
                    'length_before_control': 0.98,
                    'control_width': 0.02,
                    'mesh': './backward_step.h5',
                    'step_height': 0.1,
                    }

flow_params = {'mu': 1E-3,
                'rho': 1,
                'inflow_profile': profile
                }

solver_params = {'dt': 5E-4,
                 'solver_type': 'lu',
                 'preconditioner_step_1': 'hipre_amg',
                 'preconditioner_step_2': 'hipre_amg',
                 'preconditioner_step_3': 'hipre_amg',
                 'la_solver_step_1': 'cg',
                 'la_solver_step_2': 'cg',
                 'la_solver_step_3': 'cg'
                 }

U_in = 1.5
Re = flow_params['rho']*U_in*geometry_params['step_height']/flow_params['mu'] 
print("Reynolds = ", Re)

# Use direct solver as archetype to compare other performances

solver = FlowSolver(flow_params, geometry_params, solver_params)

root_u = 'test_cfd/u_'
root_p = 'test_cfd/p_'

xdmf_fileu = XDMFFile(root_u+"{}.xdmf".format('lu'))
xdmf_filep = XDMFFile(root_p+"{}.xdmf".format('lu'))

n=1000

start = time.time()

for i in range(n):
    u_lu, p_lu = solver.evolve([1, 10])
    # print(i)
    # u_m=u.vector().norm('l2')/len(u.vector())
    # p_m=p.vector().norm('l2')/len(p.vector())
    # print('|u|= %g' % u_m, '|p|= %g' % p_m)
    if i == n-1:
        xdmf_fileu.write(u_lu, i*solver_params['dt'])
        xdmf_filep.write(p_lu, i*solver_params['dt'])

end = time.time()

seconds= end - start

print("Seconds required ", seconds)

best_time = seconds
best_combination = 'lu'

with open("test_cfd/compare_solvers.csv", "w") as csv_file:
    writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
    writer.writerow(["Combination", "Norm of differenc w/ lu for velocity", "and for pressure", "time needed"])



xdmf_fileu.close()
xdmf_filep.close()

solver_params['solver_type']='la_solve'

for solver1 in list_of_solvers:
    solver_params['la_solver_step_1']=solver1
    for solver2 in list_of_solvers:
        solver_params['la_solver_step_2']=solver2
        for solver3 in list_of_solvers:
            solver_params['la_solver_step_3']=solver3
            for precond1 in list_of_preconditioners:
                solver_params['preconditioner_step_1']=precond1
                for precond2 in list_of_preconditioners:
                    solver_params['preconditioner_step_2']=precond2
                    for precond3 in list_of_preconditioners:
                        solver_params['preconditioner_step_3']=precond3

                        solver = FlowSolver(flow_params, geometry_params, solver_params)

                        combination ="{}_{}_{}_{}_{}_{}".format(solver1, precond1, solver2, precond2, solver3, precond3)
                        
                        # xdmf_fileu = XDMFFile(root_u + combination + ".xdmf")
                        
                        start = time.time()

                        for i in range(n):
                            u_, p_ = solver.evolve([1, 10])
                            # if i == n-1:
                                # print(p_.vector().norm('l2'))

                                # xdmf_fileu.write(u_, i*solver_params['dt'])
                            #     xdmf_filep.write(p_, i*solver_params['dt'])

                        end = time.time()

                        # xdmf_fileu.close()

                        seconds = end - start

                        u_err=(u_.vector()-u_lu.vector()).norm('l2')/len(u_.vector())
                        p_err=(p_.vector()-p_lu.vector()).norm('l2')/len(p_.vector())
                        print(combination + " err_u = {}, err_p = {}".format(u_err,p_err))
                        print("Seconds required ", seconds)

                        with open("test_cfd/compare_solvers.csv", "a") as csv_file:
                            writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                            writer.writerow([combination, u_err, p_err, seconds])

                        if seconds < best_time and u_err+p_err < 1e-1:
                            best_time = seconds
                            best_combination = combination

print("\n\nBest model found " + best_combination + " time of simualtion: {}".format(best_time))

                        

# xdmf_fileu.parameters["flush_output"] = True

# while True:
# for i in range(5000):

#     u, p = solver.evolve([1, 10])
#     print(i)
#     u_m=u.vector().norm('l2')/len(u.vector())
#     p_m=p.vector().norm('l2')/len(p.vector())
#     print('|u|= %g' % u_m, '|p|= %g' % p_m)

    # hdf.write(u, 'velocity', i)
    # xdmf_fileu.write(u, i*solver_params['dt'])
    # xdmf_filep.write(p, i*solver_params['dt'])
    


# evaluate solvers and preconditioners


