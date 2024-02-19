from dolfin import *
import numpy as np


class JetBCValue(UserExpression):
    '''
    Implements Boundary condition function for velocity
    onto our jet, taking as input frequency of the impulse and amplitude (Q) that are
    provided from the aget at each actuation

    Length_before_control, step height and control_width are the physical parameters
    necessary to evaluate the function at the physical point x = (x[0], x[1]) of the jet BC
    '''
    def __init__(self, time, length_before_control, control_width, step_height, a1, a2, b, frequency, Q,control_length,**kwargs):
        
        # frequancy of the sinusoidal profile
        self.freq = frequency
        
        self.time = time
        
        # length of the segment that lies between the inlet and the jet (9.8 in our case)
        self.total_length = length_before_control + control_width
        
        self.step_height = step_height

        self.b = b
        
        self.control_length_size =  b + 1/20 * control_length

        # ampltude of the jet control (can be also negative for suction)
        self.Q = Q

        #tuning parameters for individuating the best min_max interval for our jet and our frequency
        self.a1 = a1
        self.a2 = a2
        
        
        super().__init__(self, **kwargs) 

    def eval(self, values, x):

        x_min = self.total_length - self.control_length_size
        
        y = self.step_height   
        
        x_max = x_min + self.control_length_size

        # evaluation of the profile at point x
        values[0] = self.a1*self.Q*(x[0]-x_min)*(x_max-x[0])/self.control_length_size**2 * abs(sin((self.a2*(self.freq) + self.a2/2)*2*pi*self.time))
        values[1] = self.a1*self.Q*(x[0]-x_min)*(x_max-x[0])/self.control_length_size**2 * abs(sin((self.a2*(self.freq) + self.a2/2)*2*pi*self.time))

    def value_shape(self):
        return (2, )
