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
                           'hypre_parasails', 'icc', 'ilu', 'jacobi', 'none', 'sor']

too_expensive_1_2= ['richardson']
too_expensive_2_solver= ['richardson', 'default']


too_expensive_2 = ['gmres_ilu', 'gmres_icc', 'gmres_default', 'gmres_jacobi','gmres_none', 'gmres_sor',
                   'tfqmr_default', 'tfqmr_icc', 'tfqmr_ilu', 'tfqmr_jacobi', 'tfqmr_none', 'tfqmr_sor']


too_expensive_3 = ['bicgstab_none', 'richardson_jacobi', 'richardson_none', 'richardson_sor']



geometry_params = {'frequency': 10,
                    'length_before_control': 0.98,
                    'control_width': 0.02,
                    'mesh': 'simulation_base/mesh/our_mesh.h5',
                    'step_height': 0.1,
                    }

flow_params = {'mu': 1E-3,
                'rho': 1,
                'inflow_profile': profile
                }

solver_params = {'dt': 5E-4,
                 'solver_type': 'lu',
                 'preconditioner_step_1': 'hypre_amg',
                 'preconditioner_step_2': 'hypre_amg',
                 'preconditioner_step_3': 'hypre_amg',
                 'la_solver_step_1': 'bicgstab',
                 'la_solver_step_2': 'cg',
                 'la_solver_step_3': 'cg'
                 }


flow_params['u_init'] = 'mesh/u_init.xdmf'
flow_params['p_init'] = 'mesh/p_init.xdmf'



U_in = 1.5
Re = flow_params['rho']*U_in*geometry_params['step_height']/flow_params['mu'] 
print("Reynolds = ", Re)

# Use direct solver as archetype to compare other performances

solver = FlowSolver(flow_params, geometry_params, solver_params)


root_u = 'test_cfd/u_'
root_p = 'test_cfd/p_'

xdmf_fileu = XDMFFile(root_u+"{}.xdmf".format('lu'))
xdmf_filep = XDMFFile(root_p+"{}.xdmf".format('lu'))

n=200


# ------------------------LU-simuation
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

best_time = 1000000
best_combination = 'lu'


xdmf_fileu.close()
xdmf_filep.close()

solver_params['solver_type']='la_solve'

sol1=['bicgstab',
    'cg',
    'default',
    'default',
    'gmres',
    'gmres']
pre1=['default', 'default', 'default', 'ilu', 'default', 'ilu']


sol2=['cg', 'cg', 'cg', 'gmres', 'gmres']
pre2=['amg', 'hypre_amg', 'hypre_euclid', 'hypre_parasails', 'amg', 'hypre_parasails']

sol3 = ['default', 'minres', 'minres', 'minres', 'cg']
pre3= ['jacobi', 'default', 'ilu', 'jacobi', 'jacobi']



with open("test_cfd/compare_3.csv", "a") as csv_file:
    writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
    writer.writerow(["Combination3", "Norm of differenc w/ lu for velocity", "and for pressure", "time needed"])



for solver1, precond1 in zip(sol1, pre1):
    solver_params['la_solver_step_1']=solver1
    solver_params['preconditioner_step_3']=precond1
    for solver2, precond2 in zip(sol2, pre2):
        solver_params['la_solver_step_2']=solver2
        solver_params['preconditioner_step_2']=precond2
        for solver3, precond3 in zip(sol3, pre3):
            solver_params['la_solver_step_3']=solver3
            solver_params['preconditioner_step_3']=precond3
                       
            combination ="{}_{}_{}_{}_{}_{}".format(solver1, precond1, solver2, precond2, solver3, precond3)
            # combination3 ="{}_{}".format(solver3, precond3)

            # xdmf_fileu = XDMFFile(root_u + combination + ".xdmf")
            
            start = time.time()

            for i in range(n):
                u_, p_ = solver.evolve([1, 10])
        
            end = time.time()

            seconds = end - start

            u_err=(u_.vector()-u_lu.vector()).norm('l2')/len(u_.vector())
            p_err=(p_.vector()-p_lu.vector()).norm('l2')/len(p_.vector())
            print(combination + " err_u = {}, err_p = {}".format(u_err,p_err))
            print("Seconds required ", seconds)

            with open("test_cfd/compare_3.csv", "a") as csv_file:
                writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                writer.writerow([combination, u_err, p_err, seconds])

            if seconds < best_time and u_err+p_err < 1e-1:
                best_time = seconds
                best_combination = combination

print("\n\nBest model found step " + best_combination + " time of simualtion: {}".format(best_time))

with open("test_cfd/compare_3.csv", "a") as csv_file:
    writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
    writer.writerow([best_combination, "best combination"])


