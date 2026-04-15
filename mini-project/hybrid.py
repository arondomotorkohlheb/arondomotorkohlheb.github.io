import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

from controller import Controller

'''
This script contains the hybrid model, there is a separate file for the controller to maintain clarity

'''

constants = {   'Cr': 800000,
                'C2': 200000,
                'Cb': 2000000,
                'R2': 0.02,
                'RB': 5, # resistances in the circuit
                'R1': 0.01,
                'T1': 275,
                'T0_ave': 299.15,
                'T0_delta': 10,
                'R0hat':  1/250,
                'Rhat': 0.1,
                'rho': 1000, # kg/m^3
                'g': 9.81,
                'l': 1.4, # m - the height of the window
                'omega0': 0.000002, # l/s
                'omega1': 1, # l/s
                'At': 0.01, # m^2
                'Ac': 0.01, # m^2
                'm' : 1, # kg- mass of the window
                'R0': 500000, # resitance of the pipe for the window mechanism
                'Rt': 5000000, # resistance of exit pipe of tank 2
                'dt_big': 10, # resolution of the time steps
                'dt_small':1, # for higher and lower resolution
                'Amp': 400, # amplitude of the solar radiation
                'omega_rad': 2*np.pi/(2*24*60*60), # 2 pi frequency radiation with 2T = one day
                'As' : 30, # m^2 solar panel aera
                'mu' : 0.1, # efficiency of solar panel
                'P0': 10**5, #ambient pressure
                't1' : 6*60*60, # start of the sun shining
                't2': 20*60*60, # end of the sin shining
                'T_ll': 301.15, # T lowerbar for the measurement signal of Tr
                'T_ref': 302.15,# T upper limit for the measurement signal of Tr
                'T_ul':303.15, # T reference for same purpose as above
                'T_v0':10,        # turning time of v0 [s]
                'T_v1v2': 10,         # turning time of v1 and v2
                'sim_length': (1,0,0,0) , # simulation length in days, hours, minutes and seconds
                'f1': 0.1 * 1480} # omega1*water's specific heat capacitance for 1 liter
                

#x = [Tr, T2, Tb, theta, theta dot, h]

class simulation():
    def __init__(self, valve0, valve1, x0, constants = constants) -> None:
        self.valve1 = valve1
        self.valve0 = valve0
        self.x = x0
        self.t = 0
        self.Wmode = 0
        for key in constants.keys():
            setattr(self, key , constants[key])

        self.T = self.t2-self.t1
        self.omega = np.pi/self.T # half a period so I is always positive and maximized around noon
        self.controller = Controller(self.measurement_signal()[0])
        self.sim_time = ((self.sim_length[0]*24 + self.sim_length[1])*60 + self.sim_length[2])*60+ self.sim_length[3]
        self.dxdt_ = np.matmul(self.A(), self.x) + np.matmul(self.B(), self.u()) + self.Con()

        # collecting the data for the plots        
        self.Tdata = np.empty((0,4))
        self.th = np.array([])
        self.thetas = np.array([])
        self.hs = np.array([])
        self.Wms = np.array([])
        # this part is to plot nodes of the controller so 2 point sections will be plotted where the
        # dictionary collects the start and end point for each phase
        self.control_states = {i+1:[] for i in range(10)}
        self.control_states[self.controller.node.id].append([self.t/60/60])

        # changing modes probably every 10 mintes
        self.random_array = np.random.randint(0,30, int(self.sim_time/(10*60))+1)
        # print(np.count_nonzero(self.random_array > 20))
        # exit()
        self.time_stamps_for_Wmode_change = np.array([k*10*60 for k in range(int(self.sim_time/(10*60))+1)])
        self.time_stamp_step = 0
        # print(self.time_stamps_for_Wmode_change[-1]/(60*60))
        # exit()
        self.append_data()
 
    def R(self):
        return self.Rhat * (1 - np.sin(2*self.x[3])) + self.R0hat
    
    @property
    def T0(self):
        return self.T0_ave + self.T0_delta * np.sin(2*np.pi*self.t/(24*60*60)-np.pi/2)

    def A(self): # conventional A matrix
        A = np.zeros((6,6))

    # always present terms
        #temparature
        A[0, 0] = -1/self.R2/self.Cr - 1/self.R()/self.Cr
        A[0, 1] = 1/self.R2/self.Cr
        A[1, 0] = 1/self.R2/self.C2
        A[1, 1] = -1/self.R2/self.C2

        #window

        #tank2
        A[5,5] = -self.rho*self.g/self.Rt/self.At

    #valve dependent terms
        if self.valve1 == 1: # heating
            A[1,2] = self.f1/self.C2 # boiler and radiator exchanging heat
            A[2,1] = self.f1/self.Cb
            A[2,2] = -self.f1/self.Cb
            A[1, 1] = -1/self.R2/self.C2 - self.f1/self.C2
        elif self.valve1 == -1: # cooling
            pass
            A[1, 1] = -1/self.R2/self.C2 - self.f1/self.C2

        
        if self.valve0 == 1: # window opening
            pass
        elif self.valve0 == -1: # window closing
            A[3,4] = 1
            A[4,4] =  - 1/(self.l**2/4*self.m)*self.Ac**2*self.R0*self.l/4*np.cos(np.pi/4+self.x[3]/2)*np.sin(np.pi/4-self.x[3]/2)
            # print(self.x[4])
            pass
        else:
            pass

        return A
    
    def Con(self): # constant additional terms
        Con = np.zeros(6)
        Con[0] = (self.T0/self.R() + self.Wm())/self.Cr # cooling of Tr through the window + random electric heater
        if self.valve1 == 1: # heating
            pass
        elif self.valve1 == -1: # cooling
            Con[1] = self.T1*self.f1 / self.C2 # change in T2 due to cooling with T1
            Con[5] = self.omega1/self.At # filling of tank 2
        

        if self.valve0 == 1: # window opening
            # print('yo', self.measurement_signal()[1], self.valve0)
            Con[3] = 4*self.omega0/(self.Ac*self.l*np.cos(np.pi/4+self.x[3]/2))
        elif self.valve0 == -1: # window closing
            Con[4] = 1/(self.l**2/4*self.m)*(-self.l/2*self.m*self.g*np.sin(self.x[3]))
            # print('con', Con[4])
            
        elif self.valve0 == 0:
            pass
        
        return Con
        
    def I(self) -> float: # based on sun
        if (self.t > self.t1 and self.t < self.t2):
            return self.Amp*np.sin(self.omega * (self.t-self.t1))
        elif (self.t-24*60*60 > self.t1 and self.t-24*60*60 < self.t2):
            return self.Amp*np.sin(self.omega * (self.t-self.t1-24*60*60))
        else:
            return 0
    
    def u(self): #input
        return np.array([self.I()])
        
    def B(self): # B matrix
        B = np.zeros((6, 1))
        if self.x[2] < (274+110):
           B[2,0] = self.As*self.mu * self.RB / (self.RB + self.R1) /self.Cb
        return B

    def Wm(self):
        return 0
        if  self.Wmode == 0:
            return 0
        if  self.Wmode == 1:
            return 500
        if  self.Wmode == 2:
            return 1000

    def Wmode_update(self):
        if self.t > self.time_stamps_for_Wmode_change[self.time_stamp_step]:
            # print(self.time_stamp_step, self.t/10/60, self.t/60/60)
            if self.Wmode == 0:
                if self.random_array[self.time_stamp_step] < 15:
                    temp = 0
                else:
                    temp = 1
            elif self.Wmode == 1:
                if self.random_array[self.time_stamp_step] < 12:
                    temp = 0
                elif self.random_array[self.time_stamp_step] < 24:
                    temp = 1
                else:
                    temp = 2            
            elif self.Wmode == 2:
                if self.random_array[self.time_stamp_step] < 20:
                    temp = 1
                else:
                    temp = 2
            self.Wmode = temp
            self.time_stamp_step += 1
            return 1
        else:
            return 0
            
    @property
    def dt(self):  # for the heat exchange it is a waste to have a fine resolution, 
    #but for the window and tank 2 it makes sense to have tinier time steps
        if self.valve0 == 0 and np.abs(self.dxdt[5]) < 0.01: # if window is not opening/closing and tank2 level is stable
            return self.dt_big
        else:
            return self.dt_small

    def tplus(self):
        if self.t < 0:
            raise ValueError
        self.t = self.t + self.dt

    @property
    def dxdt(self):
        return self.dxdt_

    def update_states(self):
        self.dxdt_ = np.matmul(self.A(), self.x) + np.matmul(self.B(), self.u()) + self.Con()
        self.x = self.x + self.dxdt_*self.dt
        if self.x[3] > np.pi/4:
            self.x[3] = np.pi/4
        
        elif self.x[3] < 0:
            self.x[3] = 0

        if self.x[5] < 0:
            self.x[5] = 0

    def measurement_signal(self):
        if self.x[0] < self.T_ll:
            Tr = 0
        elif self.x[0] < self.T_ref:
            Tr = 1
        elif self.x[0] < self.T_ul:
            Tr = 2
        else:
            Tr = 3
        
        if self.valve0 == 1:
            v0 = 1
        elif self.valve0 == -1:
            v0 = -1
        else:
            v0 = 0
        
        if self.valve1 == 1:
            v1, v2 = 1, 1
        elif self.valve1 == -1:
            v1, v2 = -1, -1
        else:
            v1, v2 = 0, 0
        
        if self.x[3] == np.pi/4:
            w = 2
        elif self.x[3] < 0.000001:
            self.x[3] = 0
            w = 0
        else:
            w = 1
        
        return Tr, v0, v1, v2, w
    
    def processing_control_signal(self, control_signal):
        self.valve0 = self.valve0 + control_signal[0]*self.dt/self.T_v0
        if self.valve0 > 1:
            self.valve0 = 1
            
        elif self.valve0 < -1:
            self.valve0 = -1
            

        self.valve1 = self.valve1 + control_signal[1]*self.dt/self.T_v1v2
        if self.valve1 > 1:
            self.valve1 = 1
        elif self.valve1 < -1:
            self.valve1 = -1        

    def control_node_update(self):
        # first we check if the controller should move to the next point with the current measurements
            # if yes it is printed and appended so it can be plotted later
        while self.controller.update(input = self.measurement_signal()):
            # print(int(self.t/60/60), '|', self.measurement_signal(), '->',self.controller.node.id, '->', self.controller.output)
            self.control_states[self.controller.node.id].append([self.t/60/60]) # start of new phase
            self.control_states[self.controller.previous_node.id][-1].append(self.t/60/60) # end of the previous phase
        return None

    def append_data(self):
        self.Tdata = np.vstack((self.Tdata, np.append(self.x[:3],self.T0)))
        self.th = np.append(self.th, (self.t/60/60))
        self.thetas = np.append(self.thetas, self.x[3])            
        self.hs = np.append(self.hs, self.x[5])
        self.Wms = np.append(self.Wms, self.Wm())
        return None

    def plot(self):
        fig = plt.figure(figsize=(8, 6.5))
        plot1 = plt.subplot2grid((4, 3), (0, 0), rowspan=4, colspan=2) 
        plot2 = plt.subplot2grid((4, 3), (0, 2)) 
        plot3 = plt.subplot2grid((4, 3), (1, 2)) 
        plot4 = plt.subplot2grid((4, 3), (2, 2))
        plot5 = plt.subplot2grid((4, 3), (3, 2)) 

        plot1.plot(self.th, self.Tdata-274, label = ['$T_{room}$', '$T_{radiator}$', '$T_{boiler}$', '$T_{ambient}$'])
        plot1.set_title("Temperatures $[C\degree]$/  t $[h]$") 
        plot1.legend()

        plot2.plot(self.th, self.thetas*180/np.pi)
        plot2.set_title(" $\\theta$  $[\degree]$ / t $[h]$") 
        
        plot3.plot(self.th, self.hs/1000)
        plot3.set_title("h $[m]$ / t [h]") 

        self.control_states[self.controller.node.id][-1].append(self.t/60/60)
        for key in self.control_states:
            for section in self.control_states[key]:
                plot4.plot(section, [key,key], c = 'b')

        plot4.set_title("control nodes")

        plot5.plot(self.th, self.Wms)
        plot5.set_title("$W_m$ $[W]$ / t [h] ") 
        plt.tight_layout()
        plt.show()

        return None

    def run(self):

        # running the simulation along the time span specified
        while self.t < self.sim_time:
            self.control_node_update()
            #then we are at the correct node of the controller so we extract the control signal
            self.processing_control_signal(self.controller.output)
            self.control_node_update()
            self.Wmode_update()
            self.update_states()
            self.tplus()
            self.append_data()
        # print(int(self.t/60/60), '|', self.measurement_signal(), '->',self.controller.node.id, '->', self.controller.output)

        
        

        
        
#x = [Tr, T2, Tb, theta, theta dot, h]
x0 = np.array([288, 289, 274+90, 0*np.pi/180, 0, 1])
sim = simulation(valve0=0, valve1=1, x0=x0)

sim.run()
sim.plot()



