from dolfin import *
import numpy as np


def normalize_angle(angle):
    '''Make angle in [-pi, pi]'''
    assert angle >= 0 

    if angle < pi: 
        return angle
    if angle < 2*pi:
        return -((2*pi)-angle)
        
    return normalize_angle(angle - 2*pi)


class JetBCValue(UserExpression):
    '''
    Value of this expression is a vector field v(x, y) = A(theta)*e_r
    where A is the amplitude function of the polar angle and e_r is radial 
    unit vector. The field is modulated such that 
    
    1) at theta = theta0 \pm width/2 A is 0
    2) \int_{J} v.n dl = Q

    Here theta0 is the (angular) position of the jet on the cylinder, width 
    is its angular width and finaly Q is the desired flux thought the jet.
    All angles are in degrees.
    '''
    def __init__(self, frequency, time, length_before_control, step_height, control_width, Q, **kwargs):       #Q per noi è unica cosa importante
       # assert width > 0 and radius > 0 # Sanity. Allow negative Q for suction
       #  theta0 = np.deg2rad(theta0)
       #  assert theta0 >= 0  # As coming from deg to rad
        
        self.freq = frequency
        self.time = time
        self.length_before_contr = length_before_control
        self.step_height = step_height
        self.control_width = control_width
      # self.width = np.deg2rad(width)
      # From deg2rad it is possible that theta0 > pi. Below we habe atan2 so 
      # shift to -pi, pi
      # self.theta0 = normalize_angle(theta0)

        self.Q = Q
        super().__init__(self, **kwargs)   #classe superiore

    def eval(self, values, x):
        x_min = self.length_before_contr
        y = self.step_height   
        x_max = x_min + self.control_width
            
        values[0] = self.Q*(x[0]-x_min)*(x_max-x[0])/self.control_width**2 * abs(sin(self.freq*2*pi*self.time))
        values[1] = self.Q*(x[0]-x_min)*(x_max-x[0])/self.control_width**2 * abs(sin(self.freq*2*pi*self.time))


    # This is a vector field in 2d
    def value_shape(self):
        return (2, )

# ------------------------------------------------------------------------------


if __name__ == '__main__':
    from xii import EmbeddedMesh
    from generate_msh import generate_mesh
    from msh_convert import convert
    import os

    root = 'test_jet_bcs'
    positions = [0, 60, 120, 180, 240, 300]
    geometry_params = {'jet_positions': positions,
                       'output': '.'.join([root, 'geo']),
                       'jet_width': 10,
                       'clscale': 0.25}

    h5_file = '.'.join([root, 'h5'])
    # Regenerate mesh?
    if True: 
        generate_mesh(geometry_params, template='./geometry_2d.template_geo')
        msh_file = '.'.join([root, 'msh'])
        assert os.path.exists(msh_file)

        convert(msh_file, h5_file)
        assert os.path.exists(h5_file)

    comm = mpi_comm_world()
    h5 = HDF5File(comm, h5_file, 'r')
    mesh = Mesh()
    h5.read(mesh, 'mesh', False)

    surfaces = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    h5.read(surfaces, 'facet')

    positions = [0, 60, 120, 180, 240, 300]

    cylinder_noslip_tag = 4
    first_jet = cylinder_noslip_tag+1
    njets = len(positions)

    width = 10
    radius = 10    

    tagged_positions = zip(range(first_jet, first_jet + njets), positions)

    for tag, theta0 in tagged_positions:
        tag, theta0 = tagged_positions[0]

        cylinder = EmbeddedMesh(surfaces, markers=[tag])
        
        x, y = SpatialCoordinate(cylinder)
    
        # cylinder_surfaces = cylinder.marking_function
        
        V = VectorFunctionSpace(cylinder, 'CG', 2)

        v = JetBCValue(radius, width, theta0, Q=tag, degree=5)
        v.Q = 2*tag

        f = interpolate(v, V)
        # Outer normal of the cylinder
        n = as_vector((x, y))/Constant(radius)
            
    # For visual check
    # File('foo_%d.pvd' % tag) << f
