from cfd_solver import FlowSolver, profile
from dolfin import *
import time

import csv



# list of solvers


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

U_in = 1.5
Re = flow_params['rho']*U_in*geometry_params['step_height']/flow_params['mu'] 
print("Reynolds = ", Re)

# Use direct solver as archetype to compare other performances

solver = FlowSolver(flow_params, geometry_params, solver_params)

root_u = 'test_cfd/u_'
root_p = 'test_cfd/p_'

xdmf_fileu = XDMFFile(root_u+"{}.xdmf".format('lu'))
xdmf_filep = XDMFFile(root_p+"{}.xdmf".format('lu'))

n=5000

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


xdmf_fileu.close()
xdmf_filep.close()

solver_params['solver_type']='la_solve'

solver_params['la_solver_step_1']='cg'
solver_params['preconditioner_step_1']='default'

solver_params['la_solver_step_2']='gmres'
solver_params['preconditioner_step_2']='amg'

solver_params['la_solver_step_2']='gmres'
solver_params['preconditioner_step_2']='jacobi'




            
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
print(" err_u = {}, err_p = {}".format(u_err,p_err))
print("Seconds required ", seconds)
