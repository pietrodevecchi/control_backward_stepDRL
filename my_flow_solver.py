from my_jet_bcs import JetBCValue         # Ha sopra solo questa funzione
from dolfin import *
import numpy as np
# from printind.printind_function import printiv


class FlowSolver(object):
    def __init__(self, flow_params, geometry_params, solver_params):    # in Env2DCylinder, we create the flow simulation object ( both parameters instead are initialized in env.py)
    # self.flow = FlowSolver(self.flow_params, self.geometry_params, self.solver_params)                                                                         	
        mu = Constant(flow_params['mu'])              # dynamic viscosity
        rho = Constant(flow_params['rho'])            # density

        mesh_file = geometry_params['mesh']     # ---> lo trovi in Env2DCyl che viene inizializzato a partire da un file che è dentro la cartella mesh (Turek_2d qualcosa)                                                              
        # printiv(mesh_file)                      # utile per debugging a quanto ho capito

        # Load mesh with markers       
                                                                      ######
        mesh = Mesh() 
        comm = mesh.mpi_comm()                                               #
        h5 = HDF5File(comm, mesh_file, 'r')                                  #
                                                                             #
        h5.read(mesh, 'mesh', False)                                         # In queste righe c'è la costruzione della mesh attraverso funzioni di dolfin a partire da mesh_file
                                                                             #
        surfaces = MeshFunction('size_t', mesh, mesh.topology().dim()-1)     #
        h5.read(surfaces, 'facet')                                           #
                                                                        ######
        # for i in range(0, 100):
        #     print(surfaces.array()[i] )
        # These tags should be hardcoded by gmsh during generation
        inlet_tag = 1
        outlet_tag = 2
        wall_tag1 = 3 # bottom before jet
        wall_tag2 = 4 # step and bottom after step
        wall_tag3 = 5 # top wall
        control_tag = 6

        # cylinder_noslip_tag  = 4                                                            
                                                                    
                                                                        

        # Define function spaces
        V = VectorFunctionSpace(mesh, 'CG', 2)             #Classe di FEniCS    -> vedere esattamente come costruisce lo spazio funzionale
        Q = FunctionSpace(mesh, 'CG', 1)                   #Idem                -> "" ""

        # Define trial and test functions
        u, v = TrialFunction(V), TestFunction(V)
        p, q = TrialFunction(Q), TestFunction(Q)
        u_n, p_n = Function(V), Function(Q)

        gtime = 0.  # External clock                                                               
                                                                        
        # Starting from rest or are we given the initial state
        for path, func, name in zip(('u_init', 'p_init'), (u_n, p_n), ('u0', 'p0')):     # chiama path, func, name le tuple dentro il for 
            if path in flow_params:
                comm = mesh.mpi_comm()
                XDMFFile(comm, flow_params[path]).read_checkpoint(func, name, 0)         # se c'è già qualcosa, eventualmente riparte da lì
                # assert func.vector().norm('l2') > 0

        u_, p_ = Function(V), Function(Q)  # Solve into these

        dt = Constant(solver_params['dt'])
        # Define expressions used in variational forms
        U  = Constant(0.5)*(u_n + u)
        
        n  = FacetNormal(mesh)                         #  FacetNormal è una funzione in FEniCS che restituisce un oggetto vettoriale che rappresenta il normale alle facce della mesh.
        f  = Constant((0, 0))

        epsilon = lambda u :sym(nabla_grad(u))                  # function handle " di u" = sym(grad_u)

        sigma = lambda u, p: 2*mu*epsilon(u) - p*Identity(2)    # function handle "di u e p" = 2*mu*grad_u -p*I

        # Define variational problem for step 1              # u è la soluzione dello step 1 (u_n noto)
        F1 = (rho*dot((u - u_n) / dt, v)*dx                   # Derivata temporale
              + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx      # Trasporto
            #   + rho*dot(dot(u_n, nabla_grad(U)), v)*dx 
              + inner(sigma(U, p_n), epsilon(v))*dx           # Diffusione (qua c'è la differenza con C-T)
              + dot(p_n*n, v)*ds                              # Pressione ( correzione per stabilità )
              - dot(mu*nabla_grad(U)*n, v)*ds                 # Neumann U -> gradiente di U = grad (u) - grad(u_n) (E'all'outlet sostanzialmente, nel resto impongo DirichletBC per u)
              - dot(f, v)*dx)                                 # Forzante 



        # F1 = (rho*dot((u - u_n) / dt, v)*dx                   # Derivata temporale
        #       + rho*dot(dot(u_n, nabla_grad(u)), v)*dx      # Trasporto
        #       + inner(sigma(u, p_n), epsilon(v))*dx           # Diffusione (qua c'è la differenza con C-T)
        #       + dot(p_n*n, v)*ds                              # Pressione (  parte di bordo della correzione per stabilità )
        #       - dot(mu*nabla_grad(u_n)*n, v)*ds                 # Neumann U -> gradiente di U = grad (u) - grad(u_n) (E'all'outlet sostanzialmente, nel resto impongo DirichletBC per u)
        #       - dot(f, v)*dx)         
        a1, L1 = lhs(F1), rhs(F1)

        # Define variational problem for step 2                                      # p soluzione dello step 2 (p_n e u_ noti ---> u_ è u dello step 1)
        a2 = dot(nabla_grad(p), nabla_grad(q))*dx                                    # Laplaciano pressione (p-p_n viene già spezzato in lhs e rhs)
        L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/dt)*div(u_)*q*dx

        # Define variational problem for step 3                                      # u soluzione step 3 (p_ è il p di prima e u_ idem è u del primo step
        a3 = dot(u, v)*dx
        L3 = dot(u_, v)*dx - dt*dot(nabla_grad(p_ - p_n), v)*dx

        inflow_profile = flow_params['inflow_profile'](mesh, degree=2)            
        # Define boundary conditions, first those that are constant in time
        bcu_inlet = DirichletBC(V, inflow_profile, surfaces, inlet_tag)              # Funzione di Dolfin
        # No slip
        bcu_wall1 = DirichletBC(V, Constant((0, 0)), surfaces, wall_tag1)
        bcu_wall2 = DirichletBC(V, Constant((0, 0)), surfaces, wall_tag2)
        bcu_wall3 = DirichletBC(V, Constant((0, 0)), surfaces, wall_tag3)
        # bcu_cyl_wall = DirichletBC(V, Constant((0, 0)), surfaces, cylinder_noslip_tag)
        # Fixing outflow pressure
        bcp_outflow = DirichletBC(Q, Constant(0), surfaces, outlet_tag)              # Dirichlet all' Outlet per la Pressione

        # Now the expression for the jet
        # NOTE: they start with Q=0
                                                                        
        frequency = geometry_params['frequency']
        length_before_control = geometry_params['length_before_control']
        control_width = geometry_params['control_width']  
        step_height = geometry_params['step_height']   
        #radius = geometry_params['jet_radius']
        #width = geometry_params['jet_width']                                         # tutti definiti in env.py
        #positions = geometry_params['jet_positions']

        jet_tag = control_tag                                                                                   # (quindi il tag è 4)
        jet = JetBCValue(frequency, gtime, length_before_control, step_height, control_width, Q=0, degree=1)      # Qua crea l'oggetto della funzione JetBCV
        bcu_jet = DirichletBC(V, jet, surfaces,jet_tag)
    

        # All bcs objects togets
        bcu = [bcu_inlet, bcu_wall1, bcu_wall2, bcu_wall3, bcu_jet]   #tolto cilindro
        bcp = [bcp_outflow]

        As = [Matrix() for i in range(3)]
        bs = [Vector() for i in range(3)]

        # Assemble matrices
        assemblers = [SystemAssembler(a1, L1, bcu),
                      SystemAssembler(a2, L2, bcp),
                      SystemAssembler(a3, L3, bcu)]

        # Apply bcs to matrices (this is done once)
        for a, A in zip(assemblers, As):
            a.assemble(A)

        # Chose between direct and iterative solvers
        solver_type = solver_params.get('la_solve', 'lu')
        assert solver_type in ('lu', 'la_solve')

        
        if solver_type == 'lu':
            solvers = list(map(lambda x: LUSolver(), range(3)))
        else:
            solvers = [KrylovSolver('bicgstab', 'hypre_amg'),  # Very questionable preconditioner
                       KrylovSolver('cg', 'hypre_amg'),
                       KrylovSolver('cg', 'hypre_amg')]

        # Set matrices for once, likewise solver don't change in time
        for s, A in zip(solvers, As):
            s.set_operator(A)
            # if solver_type == 'lu':
            #     s.parameters['reuse_factorization'] = True
            # # Iterative tolerances
            # else:
            #     s.parameters['relative_tolerance'] = 1E-8
            #     s.parameters['monitor_convergence'] = True
            if not solver_type == 'lu':
                s.parameters['relative_tolerance'] = 1E-8
                s.parameters['monitor_convergence'] = True

        


        # Things to remeber for evolution
        self.jet = jet
        # Keep track of time so that we can query it outside
        self.gtime, self.dt = gtime, dt
        # Remember inflow profile function in case it is time dependent
        self.inflow_profile = inflow_profile

        self.solvers = solvers
        self.assemblers = assemblers
        self.bs = bs
        self.u_, self.u_n = u_, u_n
        self.p_, self.p_n= p_, p_n

        # Rename u_, p_ for to standard names (simplifies processing)
        u_.rename('velocity', '0')
        p_.rename('pressure', '0')

        # Also expose measure for assembly of outputs outside
        self.ext_surface_measure = Measure('ds', domain=mesh, subdomain_data=surfaces)

        # Things to remember for easier probe configuration
        self.viscosity = mu
        self.density = rho
        self.normal = n
        self.jet_tag = jet_tag

  
  
  
  
  # 2)
  
  
  
  
  
    def evolve(self, jet_bc_values):
        '''Make one time step with the given values of jet boundary conditions'''
        #assert len(jet_bc_values) == len(self.jets)

        # Update bc expressions
        # for Q, jet in zip(jet_bc_values, self.jets): jet.Q = Q
        self.jet.Q = jet_bc_values
        

        #MISA CHE BISOGNA PASSARGLI ANCHE IL TEMPO

        # Make a step
        self.gtime += self.dt(0)        # -----> dt è in SOLVER PARAMS

        self.jet.time = self.gtime

        inflow = self.inflow_profile
        if hasattr(inflow, 'time'):   #controlla che inflow abbia l'attributo "time" (nel  nostro caso, no)
            inflow.time = self.gtime

        assemblers, solvers = self.assemblers, self.solvers
        bs = self.bs
        u_, p_ = self.u_, self.p_
        u_n, p_n = self.u_n, self.p_n

        for (assembler, b, solver, uh) in zip(assemblers, bs, solvers, (u_, p_, u_)):
            assembler.assemble(b)
            solver.solve(uh.vector(), b)

        u_n.assign(u_)
        p_n.assign(p_)

        # Share with the world
        return u_, p_






def profile(mesh, degree):
    bot = mesh.coordinates().min(axis=0)[1]+0.1
    top = mesh.coordinates().max(axis=0)[1]
    H = top - bot
    print(bot)
    print(top)
    Um = 1.5

    return Expression(('-4*Um*(x[1]-bot)*(x[1]-top)/H/H',
                       '0'), bot=bot, top=top, H=H, Um=Um, degree=degree)

# --------------------------------------------------------------------

if __name__ == '__main__':
    geometry_params = {'frequency': 10,
                       'length_before_control': 0.98,
                       'control_width': 0.02,
                       'mesh': './backward_step.h5',
                       'step_height': 0.1,
                       }

    flow_params = {'mu': 1E-3,
                   'rho': 1,
                   'inflow_profile': profile}
    
    solver_params = {'dt': 5E-4}
    # solver_params = {'dt': 0}

    print(geometry_params)

    solver = FlowSolver(flow_params, geometry_params, solver_params)
    
    xdmf_fileu = XDMFFile("results_u.xdmf")
    xdmf_filep = XDMFFile("results_p.xdmf")

    xdmf_fileu.parameters["flush_output"] = True

    # while True:
    for i in range(5000):

        u, p = solver.evolve(1)
        print(i)
        u_m=u.vector().norm('l2')/len(u.vector())
        p_m=p.vector().norm('l2')/len(p.vector())
        print('|u|= %g' % u_m, '|p|= %g' % p_m)

        # hdf.write(u, 'velocity', i)
        xdmf_fileu.write(u, i*solver_params['dt'])
        xdmf_filep.write(p, i*solver_params['dt'])
        

    xdmf_fileu.close()
    xdmf_filep.close()

    