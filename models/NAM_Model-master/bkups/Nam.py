# region modules

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn
from scipy import stats
from matplotlib.offsetbox import AnchoredText
import matplotlib.dates as mdates
import logging

logging.basicConfig(format='%(levelname)s: %(module)s.%(funcName)s(): %(message)s')
pd.plotting.register_matplotlib_converters(explicit=True)
seaborn.set()
np.seterr(all='ignore')
import objectivefunctions as obj
import NAM_func as nm

# endregion

class Nam(object):
    _dir = r'C:\works\NileSEC\Projects\PhaseII\Codes\NAM\NAM_Model-master\NAM_Model-master'
    _data = "Test1.csv"

    def __init__(self, Area=1592, Cal=False):
        self._working_directory = None
        self.Data_file = None
        self.df = None
        self.P = None
        self.T = None
        self.E = None
        self.Qobs = None
        self.area = Area / (3.6 * 24)
        self.Area = Area
        self.Spinoff = 0
        self.parameters = None
        self.initial = np.array([10, 10, 0.5, 500, 10, 0.5, 0.5, 0, 2000, 2.15,2])
		#self.initial = np.array([1.56241373e+00,9.99990757e+02,5.19725550e-06,8.24447934e+02,4.60493449e+01,5.81233225e-01,7.43898809e-01,3.31118444e-01,1.77752369e+03,3.83664028e+00,3.98544379e+00])
        self.Qsim = None
        self.n = None
        self.Date = None
        self.bounds = ((0.01, 50), (0.01, 1000), (0.01, 1), (200, 1000), (10, 50), (0.01, 0.99), (0.01, 0.99), (0.01, 0.99), (500, 5000), (0, 4), (-2, 4))
        self.NSE = None
        self.RMSE = None
        self.PBIAS = None
        self.Cal = Cal
        self.statistics = None
        self.export = 'Result.csv'
        self.paraFiles = 'para.csv'

    @property
    def process_path(self):
        return self._working_directory

    @process_path.setter
    def process_path(self, value):
        self._working_directory = value
        pass

    def DataRead(self):
        self.df = pd.read_csv(self.Data_file, sep=',', parse_dates=[0], header=0)
        self.df = self.df.set_index('Date')

    def InitData(self):
        self.P = self.df.P
        self.T = self.df.Temp
        self.E = self.df.E
        self.Qobs = self.df.Q
        self.n = self.df.__len__()
        self.Qsim = np.zeros(self.n)
        self.Date = self.df.index.to_pydatetime()

    def nash(self, qobserved, qsimulated):
        if len(qobserved) == len(qsimulated):
            s, e = np.array(qobserved), np.array(qsimulated)
            # s,e=simulation,evaluation
            mean_observed = np.nanmean(e)
            # compute numerator and denominator
            numerator = np.nansum((e - s) ** 2)
            denominator = np.nansum((e - mean_observed) ** 2)
            # compute coefficient
            return 1 - (numerator / denominator)

        else:
            logging.warning("evaluation and simulation lists does not have the same length.")
            return np.nan

    def Objective(self, x):
        self.Qsim = nm.NAM(x, self.P, self.T, self.E, self.area, self.Spinoff)
        # n = math.sqrt((sum((self.Qsim - self.Qobs) ** 2)) / len(self.Qobs))
        n = obj.nashsutcliffe(self.Qobs, self.Qsim)
        return -n

    def run(self):
        self.DataRead()
        self.InitData()
        if self.Cal == True:
            self.parameters = minimize(self.Objective, self.initial, method='SLSQP', bounds=self.bounds,
                                       options={'maxiter': 2000, 'disp': True})
            self.Qsim = nm.NAM(self.parameters.x, self.P, self.T, self.E, self.area, self.Spinoff)
        else:
            self.Qsim = nm.NAM(self.initial, self.P, self.T, self.E, self.area, self.Spinoff)
        
    def update(self):
        self.df['Qsim'] = self.Qsim
        self.df.to_csv(os.path.join(self.process_path, self.export), index=True,header=True)
		
    def stats(self):
        mean = np.mean(self.Qobs)
        mean2 = np.mean(self.Qsim)
        self.NSE = 1 - (sum((self.Qsim - self.Qobs) ** 2) / sum((self.Qobs - mean) ** 2))
        self.RMSE = np.sqrt(sum(self.Qsim - self.Qobs) ** 2) / len(self.Qsim)
        self.PBIAS = (sum(self.Qobs - self.Qsim) / sum(self.Qobs)) * 100
        self.statistics = obj.calculate_all_functions(self.Qobs, self.Qsim)

    def draw(self):
        self.stats()
        width = 10  # Figure width
        height = 5  # Figure height
        f = plt.figure(figsize=(width, height))
        ax1 = f.add_subplot(211)
        ax2 = f.add_subplot(212)
        color = 'tab:blue'
        ax2.set_ylabel('Precipitation ,mm ', color=color, style='italic', fontweight='bold', labelpad=20, fontsize=13)
        ax2.bar(self.Date, self.df.P, color=color, align='center', alpha=0.6, width=1)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, max(self.df.P) * 1.1, )
        ax2.tick_params(axis='x', labelrotation=45)
        ax2.legend(['Precipitation'])
        ax2.set_xlabel('Date', style='italic', fontweight='bold', labelpad=20, fontsize=13)
        color = 'tab:red'
        ax1.set_title('NAM Simulation', style='italic', fontweight='bold', fontsize=16)
        ax1.set_ylabel(r'Discharge m$^3$/s', color=color, style='italic', fontweight='bold', labelpad=20, fontsize=13)
        ax1.plot(self.Date, self.Qobs, 'b-', self.Date, self.Qsim, 'r--', linewidth=2.0)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(('Observed Run-off', 'Simulated Run-off'))
        plt.setp(ax1.get_xticklabels(), visible=False)
        anchored_text = AnchoredText("NSE = %.2f\nRMSE = %0.2f\nPBIAS = %0.2f" % (self.NSE, self.RMSE, self.PBIAS),
                                     loc=5)
        ax1.add_artist(anchored_text)
        plt.subplots_adjust(hspace=0.05)
        f.tight_layout()
        plt.show()

# Initilize object
Nam = Nam(Area=1592, Cal=True)
# Process path
Nam.process_path = r'C:\works\NileSEC\Projects\PhaseII\Codes\NAM\NAM_Model-master\NAM_Model-master'
# Data file
Nam.Data_file = os.path.join(Nam.process_path, "Test1.csv")
Nam.run()
Nam.draw()
Nam.update()
