from scipy.optimize import curve_fit
import pandas as pd
import numpy as np

import os
import glob
import re
import sys
import logging 
import inspect
import json

import matplotlib.pyplot as plt

#setting up basic logging
logging.basicConfig(filename="log.txt",
                    filemode='w',
                    format='(%(asctime)s,%(msecs)d)%(name)s ---> %(levelname)s : %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

def log_run():
    curframe=inspect.currentframe()
    frame=inspect.getouterframes(curframe, 3)

    logging.debug(f"Run the {frame[1][3]}")


#defining main calculation class
class FittingClass:
    def __init__(self, params):
        self.params = params
        self.result = []
        self.current_file = ''
        self.beaten_files = []

    @staticmethod
    def fit_simple_sine_decay(x, Amp, Dec, Freq, Phase, Zero):
        return Amp*np.sin(2*np.pi*Freq*x + Phase)*np.exp(-Dec*x) + Zero

    def ReadDataFolder(self):
        """
        This function gets all unique filenames in data folder
        """
        log_run()

        try:
            filenames = list(set([x.split('\\')[1][:-4] for x in glob.glob('data/*')]))
            if len(filenames) == 0:
                raise Exception("No filenames")
            self.filenames = filenames
            
            logging.debug(filenames)
            return filenames
        except Exception as e:
            logging.critical(f"Can't read the data folder")
            logging.error(e)
            raise
    
    def ReadDataFile(self, filename):
        """
        This function reads .dat file
        """
        log_run()

        logging.debug(f'Reading the {filename}.dat')
        try:
            data_from_file = np.loadtxt(f'{filename}.dat', delimiter ='\t')
        except Exception as e:
            logging.error(f'Can\'t read {filename}.dat')
            logging.error(e)
            raise

        data = pd.DataFrame({
            "Delay, mm" : data_from_file[:, 0],
            "Signal, mV" : data_from_file[:, 1],
        })
        logging.debug(f'data is \n {data}')

        return data

    def GetOverlap(self, filename):
        """
        This function collects overlap position from .txt file, or uses predefined one
        """
        log_run()

        if self.params['AutoFindOverlap']:
            try:
                with open(f"{filename}.txt", 'r') as file:
                    text = file.read()
            except Exception as e:
                logging.error(f"Failed to read {filename}.txt")
                raise

            text_lines = text.lower().split("\n")
            
            for line in text_lines:
                if "overlap" in line:
                    match = re.search('\d{,3}\.\d{,3}', line)[0]
                    if match:
                        try:
                            new_overlap = float(match)
                        except Exception as e:
                            logging.error(f'Can\'t make float from {match}')
                            logging.error(e)
                            new_overlap = self.params['zero_position']
                            logging.debug(f'Using predefined overlap at {self.params["zero_position"]}')
                            break
                        logging.debug(f'Found overlap at {new_overlap}')
                    break
            
            overlap = new_overlap
        else:
            overlap = self.params['zero_position']
            logging.debug(f'Using predefined overlap at {overlap}')

        return overlap

    def ProcessData(self, data, overlap):
        """
        This function is used to convert some fields in data
        """
        log_run()

        # ns = mm*2|'forth and back'|/(2.99*10**11 |'speed of light in mm/s'|)*10**9 |'cast s to ns'| = mm *2/299
        data["Delay, ns"] = (data["Delay, mm"] - overlap)*2/299 

        if len(np.array(data[data["Delay, ns"]<-0.01]["Signal, mV"])) > 0:
            zero_level = np.average(np.array(data[data["Delay, ns"]<-0.01]["Signal, mV"]))
            data["Signal, mV"] = data["Signal, mV"] - zero_level
            logging.debug('Data is processed successfully')
        else:
            logging.debug('Failed to substract background')

        return data

    def FitProcess(self, data):
        """
        This function is responsible for fitting data with curve_fit
        """
        log_run()

        x = data[data["Delay, ns"] > 0.01]["Delay, ns"]
        y = data[data["Delay, ns"] > 0.01]["Signal, mV"]
        
#Fitting bounds
        bounds =([0, 0.01, 0.01, 0, -1e3], [1e3, 10, 10, 2*np.pi, 1e3])

        #logging.debug(f'this is x {x} and y {y}')
        try:
            p_opt, p_cov = curve_fit(FittingClass.fit_simple_sine_decay, x, y, bounds=bounds)
            p_err = np.sqrt(np.diag(p_cov))
        except Exception as e:
            logging.error(f'Failed to fit')
            logging.error(e)
            raise 
            return None, None, None
        

        parameters = {
            "Amplitude" : p_opt[0],
            "Decay" : p_opt[1],
            "Frequency" : p_opt[2],
            "Phase" : p_opt[3],
            "Zero" : p_opt[4],
            "A_err" : p_err[0],
            "D_err" : p_err[1],
            "F_err" : p_err[2],
            "P_err" : p_err[3],
            "Z_err" : p_err[4] 
        }

        Eps = np.sqrt(sum([x**2 for x in p_err/p_opt]))

        return parameters, Eps, p_opt

    def Plotting(self, data, p_opt, Eps):
        """
        This function prepeares Bokeh plot
        """
        log_run()

        try:
            plt.xlabel('Delay (ns)')
            plt.ylabel('Signal (mV)')
            plt.title(f"Series {self.current_file[:30]}")
            plt.grid(which='major', color='black')
            plt.minorticks_on()
            plt.grid(which='minor', color='lightgrey')
            plt.scatter(data['Delay, ns'], data['Signal, mV'], s=3, c='green', label='Experiment')
            plt.plot(data['Delay, ns'], FittingClass.fit_simple_sine_decay(data['Delay, ns'], *p_opt),
                        color='red', linewidth=2, label=f"Fit with Eps={round(Eps, 2)}")
            plt.legend()

            plt.savefig(f"graphs/{self.current_file}.png", dpi=150)

            plt.close()

            logging.debug("Successfully saved graph")
        except Exception as e:
            logging.error(f"Can't save file for {self.current_file}")
            logging.error(e)
            raise

    def ProcessFileFit(self):
        """
        This function holds all the fitting process for single file
        """
        log_run()

        full_filename = f'data/{self.current_file}'
        data = self.ReadDataFile(full_filename)
        overlap = self.GetOverlap(full_filename)

        logging.debug(f'This is overlap {overlap}')

        data = self.ProcessData(data, overlap)
    
        parameters, Eps, p_opt = self.FitProcess(data)

        self.Plotting(data, p_opt, Eps)

        self.result.append({
            'Name' : self.current_file,
            'Overlap' : overlap,
            'Data(pandas json)' : data.to_json(), 
            'Parameters' : parameters, 
            'Eps' : Eps,
            "_p_opt" : list(p_opt)
        })

    
    def Fit_all(self):
        log_run()
        
        for filename in self.ReadDataFolder():
            try:
                logging.debug(f'Fitting the {filename}')
                self.current_file = filename
                self.ProcessFileFit()
            except Exception as e:
                logging.error(f"Can\'t process fit for {filename}")
                logging.error(e)
                logging.debug('Skipping it')
                self.beaten_files.append(filename)
                pass

        self.result.append({
            'BeatenFiles' : self.beaten_files
        })

        return self.result


def FitPumpProbe(params):
    """
    This is resulting function, main one
    """
    log_run()

    fitter = FittingClass(params)

    Result = []
    #Section for setting up parameters
    logging.debug("Setting up parameters")
    logging.debug(params)

    fitter = FittingClass(params)

    Result = fitter.Fit_all()

    return Result


logging.info(f"Running with __name__ = {__name__}")        

#Accepting data from pipeline
try:    
    logging.info("Started")
    logging.debug("Trying to read cmd")
    cmd_data = sys.argv
    logging.debug(len(cmd_data))
    logging.debug(cmd_data)
    params = None
    if len(cmd_data) > 1:
        params = json.loads(cmd_data[1])
        logging.debug(f"Params loaded successfully {params}")
except Exception as e:
    logging.critical(f"Failed to set up pipline with the main process")
    logging.error(e)

if __name__ == "__main__":

    #setUp some default parameters
    if(params is None):
        logging.info("No input parameters, default placed")

        params = {
            'zero_position' : 129,
            'AutoFindOverlap' : True
        }

        logging.debug(params)

    res = FitPumpProbe(params)

    logging.info("Finished")

    print(json.dumps(res))