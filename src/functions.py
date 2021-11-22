#great code
import csv
import os
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
from statsmodels.tsa.ar_model import AutoReg
from scipy.stats import kurtosis
from scipy.stats import skew

df = pd.read_csv(r'C:\Users\Felix\Dropbox\Courses\Year 2\Advanced Machine Learning\task2/X_train.csv', index_col='id')


def filter_signal(data):
    # input the raw dataframe
    # return filtered signal using the biospy
    N = np.shape(data)[0]
    processed = pd.DataFrame().reindex_like(data)
    for i in range(N):
        processed_i = ecg.ecg(data.loc[i].dropna().to_numpy(dtype='float32'), 300, show = False)['filtered']
        processed.iloc[i, 0:len(processed_i)] = processed_i
    return processed

def extract_rpeaks(data):
    # input the filtered data frame, extract the peaks of the heart beats recorded in ecg
    # return the r-peaks as a numpy array
    N = np.shape(data)[0]
    large = 1000 # set column numbers to some large number, do not know how many peaks
    rpeaks = np.empty([N, large]).astype(int)
    for i in range(N):
        peaks_i = ecg.engzee_segmenter(data.loc[i].dropna().to_numpy(dtype='float32'), 300)['rpeaks']
        rpeaks[i, 0:len(peaks_i)] = peaks_i
    return rpeaks

def extract_pqst_peaks(data, r_peaks):
    # input the filtered data and the r peaks, get back the p, q, s, t peaks
    # returns stacked matrix with indices indicating location of p, q, s, t wave
    N = np.shape(data)[0]
    K  = np.shape(data)[1]
    large = 1000
    keys = ['ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks']
    peaks = np.empty([len(keys), N, large]).astype(int)
    for i in range(N):
        r_i = r_peaks[i][r_peaks[i]>0]
        if len(r_i) < 5: # if insufficient number of r-peaks could be detected, cannot find t, s peaks
            #print(i)
            continue          
        _, delineated = nk.ecg_delineate(data.loc[i].dropna().to_numpy(dtype='float32'), r_i, 300)
        # delineated is a dictionary containing the indices
        for k, letter in enumerate(keys):
            # replace nans with zeros, those will later be dropped
            indices = np.nan_to_num(np.array(delineated[letter]).astype(int), nan = 0) 
            peaks[k, i, 0:len(indices)] = indices
    return peaks

def onset_qrs(signal, q):
    # signal is an array, representing the ecg signal for one patient
    # q is a 1D array, representing previously detected locations of the q-peak (or trough)
    # returns an a 1D array indicating onset of the qrs sequence
    onset_qrs = np.empty(len(q))
    for i in range(len(q)):
        until_trough = signal[0:q[i]] # signal before q trough
        change = np.diff(until_trough)
        where_positive = np.argwhere(change>0) # get indices where signal was increasing for last time
        if np.shape(where_positive)[0] == 0:
            onset_qrs[i] = 0 # if nowhere positive before just use zero for now
        else:
            onset_qrs[i] = max(where_positive)[0]+1 # get index of closest positive measurement before q trough
    return onset_qrs

def end_qrs(signal, s):
    # signal is an array, representing the ecg signal for one patient
    # q is a scalar, representing previously detected location of the q-peak (or trough)
    # returns an index that indicates the onset of qrs complex
    end_s = np.empty(len(s))
    for i in range(len(s)):
        from_trough = signal[s[i]:] # signal after s-trough
        change = np.diff(from_trough)
        where_negative = np.argwhere(change<0)
        if np.shape(where_negative)[0] == 0:
            end_s[i] = 0 # get zero if never increases again
        else:
            end_s[i] = min(where_negative)[0]+1 # get index when signal starts decreasing again
    return end_s

def get_moments(measure, name, bla):
    # compute moments from the measure and put them into dictionary bla
    key = str(name) + "median"
    bla[key] = np.median(measure)
    key = str(name) + "mean"
    bla[key] = np.mean(measure)
    key = str(name) + "sd"
    bla[key] = np.std(measure)
    key = str(name) + "kurt"
    bla[key] = kurtosis(measure)
    key = str(name) + "skew"
    bla[key] = skew(measure)
    return bla

def extract_qrs_features(data, r_peaks, peaks):
    #input the cleaned signal, the previously detected pqrst peaks
    # calculate mean, median, variance etc for each patient of
    # amplitude of the qr wave and length of the qrs corridor
    N = np.shape(data)[0]
    large = 1000
    qr_amplitudes = np.empty([N, large])
    qrs_corridor = np.empty([N, large])
    rr_dist = np.empty([N, large])
    features = []
    for i in range(N):
        moments_i = {} # put the resulting moments summarizing signal i in a dictionary
        
        # inputs are the cleaned signal, and the q, r, s peaks previously detected
        signal_i = data.loc[i].dropna().to_numpy(dtype='float32')
        # signal_i = signal_i-np.mean(signal_i) # do not deman
        r_i = r_peaks[i][r_peaks[i]>0]
        p_i = peaks[0, i, :][peaks[0, i, :]>0]
        q_i = peaks[1, i, :][peaks[1, i, :]>0]
        s_i = peaks[2, i, :][peaks[2, i, :]>0]
        t_i = peaks[3, i, :][peaks[3, i, :]>0]
        
        # compute distance between successive r-peaks and extract moments
        rr_dist = np.diff(r_i) # this could be done better in case the rr features are not successive
        rr_dist = rr_dist 
        moments_i = get_moments(rr_dist, "rr diff ", moments_i)      
        
        # compute the amplitude of the signal (as measured by difference in signal between q and r)
        # compute the length of the qrs sector in miliseconds
        if len(r_i) == len(q_i) == len(s_i):
            # for about 95 percent we have detected equal number of q and r peaks
            # impute the missing for those later somehow
            
            # compute the qr amplitude and calculate moments
            amp_qr = signal_i[r_i]-signal_i[q_i]
            moments_i = get_moments(amp_qr, "amp qr ", moments_i)
            
            # compute the length of the qrs sequence
            qrs = s_i - onset_qrs(signal_i, q_i)
            moments_i = get_moments(qrs, "qrs ", moments_i)
        
        # also just add basic moments of the entire signal
        moments_i = get_moments(signal_i, "signal ", moments_i)
        
        #compute # of detected, Q, R, S peaks normalized by length of signal
        moments_i["R peaks"] = len(r_i)/len(signal_i)
        moments_i["S peaks"] = len(s_i)/len(signal_i)
        moments_i["Q peaks"] = len(q_i)/len(signal_i)
        moments_i["T peaks"] = len(t_i)/len(signal_i)
        moments_i["P peaks"] = len(p_i)/len(signal_i)
        moments_i["P/Q ratio"] = len(p_i)/len(q_i) if len(q_i) > 0 else 0
        #add dictionary to list
        features.append(moments_i)
        
#convert list of dictionary to df
    return pd.DataFrame(features)
                  
def ar_fit(data, p):
    # to capture the autocorrelation of the signal, fit an AR(p) model to the data
    # Can think of p as tuning parameter
    coefs = []
    N = np.shape(data)[0]
    for i in range(N):
        signal_i = data.loc[i].dropna().to_numpy(dtype='float32')
        ARmodel = AutoReg(signal_i, p).fit()
        estimates = ARmodel.params
        # compute the error of the model.
        error = np.sum(np.square(signal_i[p:] - ARmodel.predict()))/(len(signal_i)-p) 
        # put into dict
        bla = {}
        for j in range(len(estimates)):
            key = "AR" + str(j)
            bla[key] = estimates[j]
        bla["AR sigma"] = error
        coefs.append(bla)
    return pd.DataFrame(coefs)

def feature_process(data, p):
    # input the raw data and the order of the AR one wants to fit
    # returns the feature matrix
    filtered_data = filter_signal(data)
    r_peaks = extract_rpeaks(filtered_data)
    other_peaks = extract_pqst_peaks(filtered_data, r_peaks)
    qrs_features = extract_qrs_features(filtered_data, r_peaks, other_peaks)
    ar_features = ar_fit(data, p) # fit ar process on the unfiltered signal, to preserve that information.
    return ar_features.join(qrs_features)

X = feature_process(df, 10)
