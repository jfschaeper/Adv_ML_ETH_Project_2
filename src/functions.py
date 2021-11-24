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

# df = pd.read_csv(r'C:\Users\Felix\Dropbox\Courses\Year 2\Advanced Machine Learning\task2/X_train.csv', index_col='id')


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
        print(i)
        r_i = r_peaks[i][r_peaks[i]>0]
        if len(r_i) < 5: # if insufficient number of r-peaks could be detected, cannot find t, s peaks
            #print(i)
            continue
        try:
        # the package has some bugs so try two methods
            _, delineated = nk.ecg_delineate(data.loc[i].dropna().to_numpy(dtype='float32'), r_i, method = "dwt", sampling_rate = 300)
        except IndexError:
            _, delineated = nk.ecg_delineate(data.loc[i].dropna().to_numpy(dtype='float32'), r_i, method = "cwt", sampling_rate = 300)
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


def frequency_features(data):
    # compute spectral density based on unfiltered data: 
    # Do not like to use the filter here since the smoothing hides the pronounced peak < 20
    # but no filter prob not ideal since the signal is not stationary
    
    N = np.shape(data)[0]
    moments =[]
    for i in range(N):
        spectrum_i = {}
        fs = 300
        _, Pxx = welch(data.loc[i].dropna().to_numpy(dtype='float32'), fs=fs, nperseg=256 , scaling="spectrum")
    
    #compute integral, and take moments from integral (indices are rough approximation)
        sum_p = np.cumsum(Pxx)
        spectrum_i["Sum spectrum < 5Hz"] = sum_p[4]
        spectrum_i["Sum spectrum 5<x<10"] = sum_p[8]-sum_p[4]
        spectrum_i["Sum spectrum 10<x<13"] = sum_p[12]-sum_p[8]
        spectrum_i["Sum spectrum 13<x<20"] = sum_p[17]-sum_p[12]
        spectrum_i["Sum spectrum 20<x<40"] = sum_p[35]-sum_p[17]
        
    # take individual spectrum values (since peak of spectral density is concentrated below 20 hz, consistent with literature)
        for j in [3, 6, 9, 12, 20, 30, 40, 50]:
            name = "Spectrum " + str(j) + " Hz"
            spectrum_i[name] = Pxx[j]
            
    # get some other moments of the spectrum, potentially irrelevant
        spectrum_i = get_moments(Pxx, "Spectrum ", spectrum_i)
        
    #append to list
        moments.append(spectrum_i)
        
    #return df
    return pd.DataFrame(moments)    

def feature_process(data, p):
    # input the raw data and the order of the AR one wants to fit
    # returns the feature matrix
    filtered_data = filter_signal(data)
    r_peaks = extract_rpeaks(filtered_data)
    other_peaks = extract_pqst_peaks(filtered_data, r_peaks)
    qrs_features = extract_qrs_features(filtered_data, r_peaks, other_peaks)
    ar_features = ar_fit(data, p) # fit ar process on the unfiltered signal, to preserve that information.
    frequencies = frequency_features(data)
    ar_features = ar_features.join(frequencies)
    return ar_features.join(qrs_features)

# X = feature_process(df, 10)

def sort_feature_names(s, orig_features):
    # sorts s based on the order in orig_features which makes sure that features that combine the same columns have the same name and can be dropped base on the name
    index = [(lambda x: orig_features.index(x))(x) for x in s.split(':')]
    zipped_lists = zip(index, s.split(':'))
    sorted_zipped_lists = sorted(zipped_lists)
    
    return ":".join([element for _, element in sorted_zipped_lists])


def feature_engineering(X, y, model, folds, deg_poly, score, n_jobs, top, path):

    crossvalidation = KFold(n_splits=folds, shuffle=True, random_state=42)
    orig_features = list(X.columns.drop('interaction', errors='ignore'))
    tested_interactions = []

    for d in range(deg_poly): # loop over the degree of polynomials
        features = list(X.columns.drop('interaction', errors='ignore'))

        # only interact the current features with the new features to avoid working twice; could be deleted
        if 'new_features' in locals(): 
            features_interact = new_features 
            num_poly = comb(len(features_interact),2) + len(features_interact)*len(features)
        else: 
            features_interact = orig_features
            num_poly = comb(len(features),2) + len(features)

        baseline = np.mean(cross_val_score(model, X, y, scoring=score, cv=crossvalidation, n_jobs=n_jobs))
        data_interactions = pd.DataFrame(index=range(X.shape[0]) , columns=range(num_poly)) # stores all interactions
        eval_interactions = pd.DataFrame(columns=['feature_A:feature_B', score], index=range(num_poly)) #  stores how good the interaction was

        i=0    
        for feature_A in features_interact: # features that are interacted with current features
            for feature_B in features:
                
                name_interaction = sort_feature_names(feature_A + ":" + feature_B, orig_features)
                
                if name_interaction not in tested_interactions: # features.index(feature_A) >= features.index(feature_B): # >= to create x^2 etc. (but x^3 is only created if x^2 has been chosen) # make sure this interaction has not been done
                    
                    print(name_interaction)
                    
                    tested_interactions.append(name_interaction) # save that this interaction has been calculated

                    X['interaction'] = X[feature_A] * X[feature_B]
                    score_eval = np.mean(cross_val_score(model, X, y, scoring=score, cv=crossvalidation, n_jobs=n_jobs))
                    
                    if score_eval > baseline: # only store new interaction if it improves on baseline
                        eval_interactions.iloc[i, : ] = pd.Series({'feature_A:feature_B' : name_interaction, score : round(score_eval,4)})
                        data_interactions.iloc[:, i] = X['interaction'] # store the good interaction data
                        data_interactions.rename(columns={i : name_interaction}, inplace=True) # give column the feature name
                        i+=1

        # check "pure" polynomials
        for feature in orig_features:

            name_interaction = ":".join([feature]*d)
            
            if name_interaction not in tested_interactions: # features.index(feature_A) >= features.index(feature_B): # >= to create x^2 etc. (but x^3 is only created if x^2 has been chosen) # make sure this interaction has not been done
                tested_interactions.append(name_interaction) # save that this interaction has been calculated

                X['interaction'] = np.power(X[feature], d)
                score_eval = np.mean(cross_val_score(model, X, y, scoring=score, cv=crossvalidation, n_jobs=n_jobs))

                if score_eval > baseline: # only store new interaction if it improves on baseline
                            eval_interactions.iloc[i, : ] = pd.Series({'feature_A:feature_B' : name_interaction, score : round(score_eval,4)})
                            data_interactions.iloc[:, i] = X['interaction'] # store the good interaction data
                            data_interactions.rename(columns={i : name_interaction}, inplace=True) # give column the feature name
                            i+=1

        # choose features and define X new
        new_features = eval_interactions.sort_values(by=score, ascending=False, ignore_index=True).loc[0:top, 'feature_A:feature_B'] # new features are the top ones of the engineered ones
        X = pd.concat([X.drop('interaction', axis=1, errors='ignore'), data_interactions[new_features]], axis=1)

    X.to_csv(path, index=False)

    return X


def engineered_testdata(X_test, features, path):
    # takes the test data and manipulates it in the same way the training data was manipulated
    n =  X_test.shape[0]
    X_test_eng = pd.DataFrame(columns=features, index=range(n))
    for feature in features:
        cols = feature.split(':')
        feat_eng = np.ones((1,n))
        for col in cols:
            feat_eng = feat_eng * np.array(X_test[col])
        X_test_eng[feature] = feat_eng.T

    X_test_eng.to_csv(path, index=False)

    return X_test_eng


