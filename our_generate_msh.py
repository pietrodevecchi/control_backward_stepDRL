import os, subprocess


'''The function generate_mesh is called in Env2DCylinder.py, as first arg are passed the geometry_params, the template is passed'''
'''The dimensionality is in-line initialized always here'''


def generate_mesh(g_parameters, template='backward_facing_step.template_geo', dim=2):     # dim stands for the dimensionality of the mesh (2D).
    '''Modify template according args and make gmsh generate the mesh'''

    # ensures that the template exists and copy the arguments passsed
    assert os.path.exists(template)
    params_cp = g_parameters.copy()

    # flag parameter to change the geometry_params
    change_parameters = 0

    with open(template, 'r') as f:
        template_lines = f.readlines()

    # keep apart the geometery_params from the rest of the template file
    template_params_code = list(map(lambda g: g.startswith('DefineConstant'), template_lines)).index(True)

    # if we want to change some parameters before meshing, we can do it here without changing the .geo file (e.g. the control width)
    if (change_parameters == 1):
        line_to_change =list(map(lambda g: g.startswith('control_width'), template_lines)).index(True)
        template_lines.pop(line_to_change)
        control_width = list(map(float, params_cp.pop('control_width')))
        control_width = 'control_width[] = {%s};\n' % (', '.join(list(map(str, control_width))))
        new_geometry_params  = ''.join([control_width] + template_lines[template_params_code:])

    # changes if necessary the 'output' (i.e. file_root) in the copy of geometry_params with the template passed here and verify that is a .geo file
    output = params_cp.pop('output')
    if not output:
      output= template
    assert os.path.splitext(output)[1] == '.geo'

    # open the template file through the variable output and change it with the changed parameters through the copy of the parameters passed in the function
    if (change_parameters == 1):
       with open(output, 'w') as f:
           f.write(new_geometry_params)
    else:
        with open(output, 'w') as f:
            f.write(' '.join(template_lines[template_params_code:]))

    #preparing the commands to run for building the mesh
    cmd = 'gmsh -0 %s ' % output
    scale = params_cp.pop('clscale')
    subprocess.call(cmd, shell=True)

    unrolled = '_'.join([output, 'unrolled'])
    assert os.path.exists(unrolled)

    #call gmesh to compute the mesh
    return subprocess.call(['gmsh -%d -format msh2 -clscale %g %s' % (dim, scale, unrolled)], shell=True)


# -------------------------------------------------------------------
# __name__ = '__main__'
if __name__ == '__main__':
    import argparse, sys, petsc4py
    from math import pi

    parser = argparse.ArgumentParser(description='Generate msh file from GMSH',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Optional output geo file
    parser.add_argument('-output', default='backward_step_ours.geo', type=str, help='A geofile for writing out geometry')
    # # Geometry
    # parser.add_argument('-length', default=200, type=float,
    #                     help='Channel length')
    # parser.add_argument('-front_distance', default=40, type=float,
    #                     help='Cylinder center distance to inlet')
    
    # # aggiungo io 
    # parser.add_argument('-coarse_distance', default=50, type=float,
    #                     help='Cylinder center distance to inlet')
    # parser.add_argument('-coarse_size', default=20, type=float,
    #                     help='Distance from the cylinder where coarsening starts')

    # parser.add_argument('-bottom_distance', default=40, type=float,
    #                     help='Cylinder center distance from bottom wall')
    # parser.add_argument('-jet_radius', default=10, type=float,
    #                     help='Cylinder radius')
    # parser.add_argument('-width', default=80, type=float,
    #                     help='Channel width')
    # parser.add_argument('-cylinder_size', default=0.5, type=float,
    #                     help='Mesh size on cylinder')
    # parser.add_argument('-box_size', default=5, type=float,
    #                     help='Mesh size on wall')
    # # Jet perameters
    # parser.add_argument('-jet_positions', nargs='+', default=[60, 120],
    #                     help='Angles of jet center points')
    # parser.add_argument('-jet_width', default=5, type=float,
    #                     help='Jet width in degrees')

    # Refine geometry
    parser.add_argument('-clscale', default=1, type=float,
                        help='Scale the mesh size relative to give')

    args = parser.parse_args()

    # Using geometry_2d.geo to produce geometry_2d.msh
    sys.exit(generate_mesh(args.__dict__))

    # FIXME: inflow profile
    # FIXME: test with turek's benchmark

    # IDEAS: More accureate non-linearity handling
    #        Consider splitting such that we solve for scalar components
