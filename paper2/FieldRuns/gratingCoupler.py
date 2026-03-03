import math
from samplemaker.devices import Device,DevicePort,registerDevicesInModule
import samplemaker.makers as sm
#from kristina_waveguides import SuspendedWaveguideSequencer, SuspendedWaveguidePort, SuspendedWaveguideConnectorOptions, SuspendedWaveguideAdapter
from waveguides import SuspendedWaveguideSequencer, SuspendedWaveguidePort, SuspendedWaveguideConnectorOptions, SuspendedWaveguideAdapter
from qplib.suspended.electrical import ElectricalConnectorOptions, ElectricalPort_PType, ElectricalPort_PType_DeepIso
import qplib.suspended.makers as qsm
import numpy as np
import samplemaker.phc as phc 
from copy import deepcopy

class grating_D(Device):
    def initialize(self): 
        self.set_name("NANOBEAM2")
        self.set_description("Same as QPLIB_FGCA, except without a port/taper on the end")
        self._seq = SuspendedWaveguideSequencer([])
        self._seq.options = SuspendedWaveguideConnectorOptions["sequencer_options"]
            
    def parameters(self):
        self.addparameter('w0',0.55,'Width of the waveguide at the start', float)
        self.addparameter('pre_split',False,'Split in quads = false', bool)
        self.addparameter('h0',2,"Holes options",int)
        
    def geom(self):
        # Grating first
       
        p = self.get_params()
      
        period = [0.6658775774440534,
                    0.6653491592155989,
                    0.6476358236213544,
                    0.6376186222129827,
                    0.648847324690748,
                    0.6875914755146024,
                    0.6449737823689897,
                    0.6600121302734347,
                    0.6760717932108108,
                    0.6635117489376464,
                    0.6561990686451578,
                    0.6673498185453912,
                    0.6531185180947873,
                    0.6531185180947873,
                    0.6531185180947873,
                    0.6531185180947873,
                    0.6531185180947873,
                    0.6531185180947873]
                
        fill_factor = [0.9235006031017126,
                        0.9105858021700755,
                        0.8915735114834373,
                        0.8561225696298673,
                        0.8337555554999322,
                        0.7773257372681367,
                        0.7457018201027427,
                        0.6864676844107501,
                        0.6504481830280842,
                        0.6372539909960407,
                        0.6608028806877202,
                        0.6669982909977046,
                        0.6646380613519955,
                        0.6900009138308456,
                        0.6613994797549103,
                        0.6597360646501718,
                        0.6434551231789021,
                        0.6944746074326602]
        
        r0 = 0
        radii = np.cumsum(period) + r0 
        
       
        w0 = p["w0"]
        h = p["h0"]
    
        g = sm.GeomGroup()  
        angle = 17.5
        R = 6.784
        for i in range(len(radii)):
            fill_a = 56/2.8

            # rX and rY (both the same here, so a circle)
            rx = (R) + (w0/2) / math.tan(math.radians(17.5))  
            ry = rx

            # waveguide width for this arc
            w = period[i] * (1 - fill_factor[i])

            # draw arc
            g += sm.make_arc(-(period[0] * (1 - fill_factor[0]))-(1/2+period[0] * (1 - fill_factor[0])),0,rx, ry, 0, w, -fill_a-1, fill_a+1,layer=3,to_poly=True,vertices=40,split=p["pre_split"])
            #g += sm.make_rect(-0.5/2,0,0.5,0.5,layer=3)
            # update radius for next arc
            if i < len(period) - 1:  
                R += (period[i] + ((period[i+1] * (1 - fill_factor[i+1])) / 2 - (period[i] * (1 - fill_factor[i])) / 2)
                )
            else:
                # last element – no i+1 available
                R += period[i] * (1 - fill_factor[i]) / 2

        
                    
    
        # waveguide
        theta = math.tan(math.radians(angle))
        port1 = SuspendedWaveguideAdapter(self._seq,"west","in")
        Gtaper = R+1
        Wtaper = w0 + 2.0 * Gtaper * theta
        
    

        g+=  sm.make_rect(R+1,0,1,3,layer=1)

       
       
        self._seq.seq = [['STATE','w',w0],
                         ["T",Gtaper,Wtaper]]
        self._seq.reset()
        g += port1.get_taper_geom(0,0,w0)
        self.addlocalport(port1.get_port())
        g += self._seq.run()
        return g


    
# Register all devices here in this module
registerDevicesInModule(__name__)