import os,sys
import numpy as np
import struct
import scipy.stats as stats
from scipy.signal import argrelextrema
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import signal
from scipy import interpolate

y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)

class pipeline(object):
    def __init__(self):
        ######## Target Sweeps ############
        #self.targ_path = raw_input('Absolute path to known good target sweep dir (e.g. /home/olimpo/olimpo_readout/sweeps/target/0805_1): ' )
	self.targ_path = '/home/lazarus/sam_git/olimpo_readout/sweeps/target/0804_3'
	data_files=[f for f in sorted(os.listdir(self.targ_path)) if f.endswith('.npy')]
        I = np.array([np.load(os.path.join(self.targ_path,f)) for f in data_files if f.startswith('I')])
        Q = np.array([np.load(os.path.join(self.targ_path,f)) for f in data_files if f.startswith('Q')])
	self.lo_freqs = np.load(self.targ_path + '/sweep_freqs.npy')
        self.raw_chan = I + 1j*Q
        #self.chan_ts /= (2**15 - 1)
	#self.chan_ts /= ((2**21 - 1) / 2048)
	self.nchan = len(self.raw_chan[0])
        self.cm = plt.cm.spectral(np.linspace(0.05,0.95,self.nchan))
        self.raw_I = self.raw_chan.real
        self.raw_Q = self.raw_chan.imag
        self.mag = np.abs(self.raw_chan)
        self.phase = np.angle(self.raw_chan)
        self.centers=self.loop_centers(self.raw_chan) # returns self.centers
	self.chan_centered = self.raw_chan - self.centers
        self.rotations = np.angle(self.chan_centered[self.chan_centered.shape[0]/2])
	self.chan_rotated = self.chan_centered * np.exp(-1j*self.rotations)
        self.phase_rotated = np.angle(self.chan_rotated)
        self.bb_freqs = np.load(self.targ_path + '/bb_freqs.npy')
        self.delta_lo = 2.5e3
        """
	prompt = raw_input('Save phase centers and rotations in ' + self.targ_path + ' (**** MAY OVERWRITE ****) (y/n)? ')
	if prompt == 'y':
		np.save(self.targ_path + '/centers.npy', self.centers)
        	np.save(self.targ_path + '/rotations.npy', self.rotations)
	"""	

    def open_stored(self, save_path = None):
        files = sorted(os.listdir(save_path))
        sweep_freqs = np.array([np.float(filename[1:-4]) for filename in files if (filename.startswith('I'))])
        I_list = [os.path.join(save_path, filename) for filename in files if filename.startswith('I')]
        Q_list = [os.path.join(save_path, filename) for filename in files if filename.startswith('Q')]
        Is = np.array([np.load(filename) for filename in I_list])
        Qs = np.array([np.load(filename) for filename in Q_list])
        return sweep_freqs, Is, Qs

    def loop_centers(self,timestream):
        def least_sq_circle_fit(chan):
            """
            Least squares fitting of circles to a 2d data set. 
            Calcultes jacobian matrix to speed up scipy.optimize.least_sq. 
            Complements to scipy.org
            Returns the center and radius of the circle ((xc,yc), r)
            """
            #x=self.i[:,chan]
            #y=self.q[:,chan]
            x=timestream[:,chan].real
            y=timestream[:,chan].imag
            xc_guess = x.mean()
            yc_guess = y.mean()
                        
            def calc_radius(xc, yc):
                """ calculate the distance of each data points from the center (xc, yc) """
                return np.sqrt((x-xc)**2 + (y-yc)**2)
    
            def f(c):
                """ calculate f, the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
                Ri = calc_radius(*c)
                return Ri - Ri.mean()
    
            def Df(c):
                """ Jacobian of f.The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
                xc, yc = c
                dfdc = np.empty((len(c), x.size))
                Ri = calc_radius(xc, yc)
                dfdc[0] = (xc - x)/Ri            # dR/dxc
                dfdc[1] = (yc - y)/Ri            # dR/dyc
                dfdc = dfdc - dfdc.mean(axis=1)[:, np.newaxis]
                return dfdc
        
            (xc,yc), success = optimize.leastsq(f, (xc_guess, yc_guess), Dfun=Df, col_deriv=True)
        
            Ri = calc_radius(xc,yc)
            R = Ri.mean()
            residual = sum((Ri - R)**2) #error in fit if needed
            #print xc_guess,yc_guess,xc,yc
            return (xc,yc),R

        centers=[]
        for chan in range(self.nchan):
                (xc,yc),r = least_sq_circle_fit(chan)
                centers.append(xc+1j*yc)
        #self.centers = np.array(centers)
        return np.array(centers)

    def plot_loop_centered(self,chan):
        plt.plot(self.chan_centered.real[:,chan],self.chan_centered.imag[:,chan],'x',color=self.cm[chan])
        plt.gca().set_aspect('equal')
        plt.xlim(np.std(self.chan_centered.real[:,chan])*-3,np.std(self.chan_centered.real[:,chan]*3))
        plt.ylim(np.std(self.chan_centered.imag[:,chan])*-3.,np.std(self.chan_centered.imag[:,chan]*3))
        plt.tight_layout()
	plt.show()
        return

    def plot_loop_rotated(self,chan):
        plt.figure(figsize = (20,20))
        plt.title('IQ loop Channel = ' + str(chan) + ', centered and rotated')
        plt.plot(self.chan_rotated.real[:,chan],self.chan_rotated.imag[:,chan],'x',color='red',mew=2, ms=6)
        plt.gca().set_aspect('equal')
        plt.xlim(np.std(self.chan_rotated.real[:,chan])*-3,np.std(self.chan_rotated.real[:,chan])*3)
        plt.ylim(np.std(self.chan_rotated.imag[:,chan])*-3,np.std(self.chan_rotated.imag[:,chan])*3)
        plt.xlabel('I', size = 20)
        plt.ylabel('Q', size = 20)
        plt.tight_layout()
	plt.show()
        return

    def multiplot(self, chan):
        #for chan in range(self.nchan):
        #rf_freqs = np.load(os.path.join(self.datapath,'light_kids.npy'))
        #rf_freq= rf_freqs[chan] - (np.max(rf_freqs) + np.min(rf_freqs))/2. + self.lo_freqs
        #print np.shape(rf_freq)
        rf_freqs = (self.bb_freqs[chan] + (self.lo_freqs/2))/1.0e6
	self.mag /= (2**15 - 1)
	self.mag /= ((2**21 - 1) / 2048)
        fig,axs = plt.subplots(1,3, figsize = (20, 12))
	axs[0].set_title('Amplitude', size = 25)
	axs[0].set_ylabel('dB', size = 25)
	axs[0].plot(rf_freqs, 20*np.log10(self.mag[:,chan]),'#8B0000', linewidth = 2)
	axs[0].tick_params(axis='y', labelsize=23) 
	axs[0].get_xaxis().set_visible(False)
	plt.grid()

	axs[1].set_title('Phase', size = 25)
	axs[1].set_ylabel('rad', size = 25)
        axs[1].plot(rf_freqs, self.phase_rotated[:,chan],'#8B0000',linewidth = 2)
	axs[1].get_xaxis().set_visible(False)
	axs[1].tick_params(axis='y', labelsize=23) 
	plt.grid()

	axs[2].set_title('I/Q loop', size = 25)
        axs[2].plot(self.chan_rotated[:,chan].real/np.max(self.chan_rotated[:,chan]),self.chan_rotated[:,chan].imag/np.max(self.chan_rotated[:,chan]),'#8B0000',marker='x', linewidth = 2)
	axs[2].get_xaxis().set_visible(False)
	axs[2].tick_params(axis='y', labelsize=23) 
        axs[2].axis('equal')                    
        plt.grid()            
        plt.tight_layout()
	plt.savefig('/home/lazarus/sam_git/multiplot.pdf', bbox = 'tight')
        plt.show()         
        return

    def plot_targ(self, path):
    	plt.ion()
	plt.figure(6)
	plt.clf()
	lo_freqs, Is, Qs = self.open_stored(path)
	lo_freqs = np.load(path + '/sweep_freqs.npy')
	bb_freqs = np.load(path + '/bb_freqs.npy')
	channels = len(bb_freqs)
	mags = np.zeros((channels,len(lo_freqs))) 
	chan_freqs = np.zeros((channels,len(lo_freqs)))
        new_targs = np.zeros((channels))
	for chan in range(channels):
        	mags[chan] = np.sqrt(Is[:,chan]**2 + Qs[:,chan]**2)
		mags[chan] /= 2**15 - 1
		mags[chan] /= ((2**21 - 1) / 512.)
		mags[chan] = 20*np.log10(mags[chan])
		chan_freqs[chan] = (lo_freqs/2 + bb_freqs[chan])/1.0e6
	mags = np.concatenate((mags[len(mags)/2:],mags[:len(mags)/2]))
	#bb_freqs = np.concatenate(bb_freqs[len(b_freqs)/2:],bb_freqs[:len(bb_freqs)/2]))
	chan_freqs = np.concatenate((chan_freqs[len(chan_freqs)/2:],chan_freqs[:len(chan_freqs)/2]))
	#new_targs = [chan_freqs[chan][np.argmin(mags[chan])] for chan in range(channels)]
	#print new_targs
	for chan in range(channels):
		plt.plot(chan_freqs[chan],mags[chan])
	plt.title('Target sweep')
	plt.xlabel('frequency (MHz)')
        plt.ylabel('dB')
	return


    def plotPSD(self, chan, time_interval):
	root_path = '/home/lazarus/sam_git/olimpo_readout/noise_measurements_0806'
	path1 = os.path.join(root_path, "1chan_roach_elec_1024_2")
	path2 = os.path.join(root_path, "500chan_avg_roach_elec")
	accum_freq = 122.
	#I = open(os.path.join(path, ""), "rb")
	#Q = open(os.path.join(path,""), "rb")
	with open(os.path.join(path1, "chP_0"), "rb") as file1:
		chan1 = file1.read()
	with open(os.path.join(path2, "chP_avg_500"), "rb") as file2:
		chan500 = file2.read()
	phases = struct.unpack(str(len(chan1)/4) + "f", chan1)
	phases500 = struct.unpack(str(len(chan500)/4) + "f", chan500)
	
	Npackets = len(phases)
        
	plot_range = (Npackets / 2) + 1
	figure = plt.figure(num= None, figsize=(20,12), dpi=200, facecolor='w', edgecolor='w')
	ax = figure.add_subplot(1,1,1)
	ax.set_xscale('log')
	ax.set_ylim((-160,-50))
        plt.xlim((0.001, accum_freq/2.))
        ax.set_ylabel('dBc/Hz', size = 28)
        ax.set_xlabel('log Hz', size = 28)
	ax.tick_params(axis='x', labelsize=25) 
	ax.tick_params(axis='y', labelsize=25) 
	plt.grid()
	
	phase_mags = np.fft.rfft(phases)
	phase_vals = (np.abs(phase_mags)**2) * (1./accum_freq) / Npackets
	phase_vals = 10*np.log10(phase_vals)
	#phase_vals -= phase_vals[0]
	
	phase_mags500 = np.fft.rfft(phases500)
	phase_vals500 = (np.abs(phase_mags500)**2) * (1./accum_freq) / Npackets
	phase_vals500 = 10*np.log10(phase_vals500)
	#phase_vals500 -= phase_vals500[0]
        
	ax.plot(np.linspace(0, accum_freq/2., (Npackets/2) + 1), phase_vals, color = 'black', linewidth = 1, label = "Single channel")
	ax.plot(np.linspace(0, accum_freq/2., (Npackets/2) + 1), phase_vals500, color = 'b', alpha = 0.5,  linewidth = 1, label = "500 channels")
	plt.legend(loc = 'upper right', fontsize = 28)
	plt.savefig('/home/lazarus/sam_git/lb_1v500.pdf', bbox = 'tight')
	plt.show()
	return

    def plot_on_off(self, chan):
	root_path = '/home/lazarus/sam_git/olimpo_readout/noise_measurements_0806'
	path_off = os.path.join(root_path, "olimpo150_phase_off_res_2")
	path_on = os.path.join(root_path, "olimpo150_phase_centered_1")
	LO_freq = (200.)*1.0e6
	accum_freq = 122.
	time_interval = 300
	#I = open(os.path.join(path, ""), "rb")
	#Q = open(os.path.join(path,""), "rb")
	with open(os.path.join(path_off, "chP_" + str(chan)), "rb") as file1:
		off = file1.read()
	with open(os.path.join(path_on, "chP_" + str(chan)), "rb") as file2:
		on = file2.read()
	phase_off = struct.unpack(str(len(off)/4) + "f", off)
	phase_on = struct.unpack(str(len(on)/4) + "f", on)
	print len(phase_off), len(phase_on)
	Npackets = len(phase_off)
        
	plot_range = (Npackets / 2) + 1
	figure = plt.figure(num= None, figsize=(20,12), dpi=200, facecolor='w', edgecolor='w')
	plt.suptitle('Chan ' + str(chan) + ' phase PSD')
	ax = figure.add_subplot(1,1,1)
	ax.set_xscale('log')
	ax.set_ylim((-150,-70))
        #plt.xlim((0.0001, self.accum_freq/2.))
        ax.set_ylabel('dBc/Hz', size = 23)
        ax.set_xlabel('log Hz', size = 23)
	ax.tick_params(axis='x', labelsize=16) 
	plt.grid()
	
	phases_on = np.fft.rfft(phase_on)
	phases_on = (np.abs(phases_on)**2 * ((1./accum_freq)**2 / (time_interval)))
	phases_on = 10*np.log10(phases_on)
	phases_on -= phases_on[0]
	phases_off = np.fft.rfft(phase_off)
	phases_off = (np.abs(phases_off)**2 * ((1./accum_freq)**2 / (time_interval)))
	phases_off = 10*np.log10(phases_off)
	phases_off -= phases_off[0]
        
	ax.plot(np.linspace(0, accum_freq/2., (Npackets/2) + 1), phases_off, color = 'black', linewidth = 1, label = "Off")
	ax.plot(np.linspace(0, accum_freq/2., (Npackets/2) + 1), phases_on, color = 'b', alpha = 0.5,  linewidth = 1, label = "On")
	plt.legend()
	plt.show()
	return

    def plot_on_off_avg(self):
	root_path = '/home/lazarus/sam_git/olimpo_readout/noise_measurements_0806'
	path_off = os.path.join(root_path, "olimpo150_phase_off_res_1")
	path_on = os.path.join(root_path, "olimpo150_phase_centered_1")
	LO_freq = (200.)*1.0e6
	accum_freq = 122.
	time_interval = 300
	phase_off = np.zeros(36621)
	phase_on = np.zeros(36621)
	for chan in range(20):
		with open(os.path.join(path_off, "chP_" + str(chan)), "rb") as file1:
			off = file1.read()
		with open(os.path.join(path_on, "chP_" + str(chan)), "rb") as file2:
			on = file2.read()
		phase_off += struct.unpack(str(len(off)/4) + "f", off)
		phase_on += struct.unpack(str(len(on)/4) + "f", on)
	
	phase_off, phase_on = phase_off/20, phase_on/20
	Npackets = len(phase_off)
        
	plot_range = (Npackets / 2) + 1
	figure = plt.figure(num= None, figsize=(20,12), dpi=200, facecolor='w', edgecolor='w')
	plt.suptitle('Chan ' + str(chan) + ' phase PSD')
	ax = figure.add_subplot(1,1,1)
	ax.set_xscale('log')
	ax.set_ylim((-150,-50))
        #plt.xlim((0.0001, self.accum_freq/2.))
        ax.set_ylabel('dBc/Hz', size = 23)
        ax.set_xlabel('log Hz', size = 23)
	ax.tick_params(axis='x', labelsize=16) 
	plt.grid()
	
	phases_on = np.fft.rfft(phase_on)
	phases_on = (np.abs(phases_on)**2 * ((1./accum_freq)**2 / (time_interval)))
	phases_on = 10*np.log10(phases_on)
	phases_on -= phases_on[0]
	phases_off = np.fft.rfft(phase_off)
	phases_off = (np.abs(phases_off)**2 * ((1./accum_freq)**2 / (time_interval)))
	phases_off = 10*np.log10(phases_off)
	phases_off -= phases_off[0]
        
	ax.plot(np.linspace(0, accum_freq/2., (Npackets/2) + 1), phases_off, color = 'black', linewidth = 1, label = "Off")
	ax.plot(np.linspace(0, accum_freq/2., (Npackets/2) + 1), phases_on, color = 'b', alpha = 0.5,  linewidth = 1, label = "On")
	plt.legend(loc = 'lower left')
	plt.show()
	phase_off = np.zeros(36621)
	return
