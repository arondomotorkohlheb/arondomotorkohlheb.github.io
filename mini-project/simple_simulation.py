import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


constants = {   'Cr': 800000,
                'C2': 200000,
                'Cb': 200000,
                'R2': 0.02,
                'RB': 5, # resistances in the circuit
                'R1': 0.01,
                'T1': 275,
                'T0': 299,
                'R0hat':  1/250,
                'Rhat': 0.1,
                'rho': 1000, # kg/m^3
                'g': 9.81,
                'l': 1.4, # m - the height of the window
                'omega0': 0.00002, # l/s
                'omega1': 0.1, # l/s
                'At': 0.01, # m^2
                'Ac': 0.01, # m^2
                'm' : 1, # kg- mass of the window
                'R0': 500000, # resitance of the pipe for the window mechanism
                'Rt': 5000000, # resistance of exit pipe of tank 2
                'dt_big': 1, # resolution of the time steps
                'dt_small':0.1, # for higher and lower resolution
                'Amp': 400, # amplitude of the solar radiation
                'omega_rad': 2*np.pi/(2*24*60*60), # 2 pi frequency radiation with 2T = one day
                'As' : 5, # m^2 solar panel aera
                'mu' : 0.1, # efficiency of solar panel
                'P0': 10**5, #ambient pressure
                't1' : 6*60*60, # start of the sun shining
                't2': 20*60*60, # end of the sin shining
                'f1': 0.01 * 1480} # omega1*water's specific heat capacitance for 1 liter
                

#x = [Tr, T2, Tb, theta, h]
x0 = np.array([290, 300, 300, 40*np.pi/180, 0, 1])

class simulation():
    def __init__(self, valve0, valve1, x0 = x0, constants = constants) -> None:
        self.valve1 = valve1
        self.valve0 = valve0
        self.x = x0
        self.t = 0
        for key in constants.keys():
            setattr(self, key , constants[key])

        self.T = self.t2-self.t1
        self.omega = np.pi/self.T # half a period so I is always positive and maximized around noon

    def R(self):
        return self.Rhat * (1 - np.sin(2*self.x[3])) + self.R0hat

    def A(self): # conventional A matrix
        A = np.zeros((6,6))

    # always present terms
        #temparature
        A[0, 0] = -1/self.R2/self.Cr - 1/self.R()/self.Cr
        A[0, 1] = 1/self.R2/self.Cr
        A[1, 0] = 1/self.R2/self.C2
        A[1, 1] = -1/self.R2/self.C2 - self.f1/self.C2

        #window

        #tank2
        A[5,5] = -self.rho*self.g/self.Rt/self.At

    #valve dependent terms
        if self.valve1 == 1: # heating
            A[1,2] = self.f1/self.C2 # boiler and radiator exchanging heat
            A[2,1] = self.f1/self.Cb
            A[2,2] = -self.f1/self.Cb
            # A[1, 1] = - self.f1/self.C2
        elif self.valve1 == -1: # cooling
            pass
            # A[1, 1] = - self.f1/self.C2

        
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
        Con[0] = self.T0/self.R()/self.Cr # cooling of Tr through the window
        if self.valve1 == 1: # heating
            pass
        elif self.valve1 == -1: # cooling
            Con[1] = self.T1*self.f1 / self.C2 # change in T2 due to cooling with T1
            Con[5] = self.omega1/self.At # filling of tank 2
        

        if self.valve0 == 1: # window opening
            Con[3] = 4*self.omega0/(self.Ac*self.l*np.cos(np.pi/4+self.x[3]/2))
        elif self.valve0 == -1: # window closing
            Con[4] = 1/(self.l**2/4*self.m)*(-self.l/2*self.m*self.g*np.sin(self.x[3]))
            # print('con', Con[4])
            
        elif self.valve0 == 0:
            pass
        
        return Con
        
    def I(self) -> float: # based on sun
        if self.t > self.t1 and self.t < self.t2:
            return self.Amp*np.sin(self.omega * (self.t-self.t1))
        else:
            return 0
    
    def u(self): #input
        return np.array([self.I()])
        
    def B(self): # B matrix
        B = np.zeros((6, 1))
        if self.valve1 == 1:
            B[2,0] = self.As*self.mu * self.RB / (self.RB + self.R1) /self.Cb
        return B
    
    @property
    def dt(self):
        if self.valve0 == 0:
            return self.dt_big
        else:
            return self.dt_small

    def tplus(self):
        if self.t < 0:
            raise ValueError
        self.t = self.t + self.dt

    def update_states(self):
        dxdt = np.matmul(self.A(), self.x) + np.matmul(self.B(), self.u()) + self.Con()
        self.x = self.x + dxdt*self.dt
        if self.x[3] >= np.pi/4:
            self.valve0 = 0
            self.x[3] = np.pi/4
        
        elif self.x[3] <= 0:
            self.valve0 = 0
            self.x[3] = 0

    def run(self):
        print(self.x)
        
        Tdata = []
        Tt = []
        thetas = []
        theta_t = []
        hs = []
        h_t = []
        thetas.append(self.x[3])
        theta_t.append(self.t)
        hs.append(self.x[5])
        h_t.append(self.t)
        while self.t < 24*60*60: # 1-day-long simulatoin
            self.update_states()
            self.tplus()
            Tdata.append(self.x[0:3])
            Tt.append(self.t/60/60)
            if np.abs(self.x[3]-thetas[-1]) > 0.001:
                thetas.append(self.x[3])
                theta_t.append(self.t)
            
            if np.abs(self.x[5]-hs[-1]) > 0.001:
                hs.append(self.x[5])
                h_t.append(self.t)
            
            

        Tdata = np.array(Tdata)
        thetas = np.array(thetas)
        theta_t = np.array(theta_t)
        hs = np.array(hs)
        h_t = np.array(h_t)
        # print(self.x)

        fig = plt.figure(figsize=(8, 6.5))
        plot1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2) 
        plot2 = plt.subplot2grid((2, 3), (0, 2)) 
        plot3 = plt.subplot2grid((2, 3), (1, 2)) 


        plot1.plot(Tt, Tdata-274, label = ['$T_{room}$', '$T_{radiator}$', '$T_{boiler}$'])
        plot1.set_title("Temperatures $[C\degree]$/  t $[h]$") 
        plot1.legend()

        plot2.plot(theta_t, thetas*180/np.pi)
        plot2.set_title(" $\\theta$  $[\degree]$ / t $[s]$") 
        
        plot3.plot(h_t, hs/100)
        plot3.set_title("h $[m]$ / t [s]") 

       
        plt.tight_layout()
        plt.show()
        plt.show()
        

# first digit is for v0 and second is for v1 and v2
# their value aligns with the measurement signal explained in the report
# for v0: 1 is opening the window, 0 idle, -1 is closing the window
# for v1, v2: 1 is heating, -1 is cooling, no idle is possible only in the hybrid code which is more comprihensive
sim = simulation(-1,1)
#print(sim.A(), sim.B())
sim.run()