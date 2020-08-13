import numpy as np

def phase_coverage(time, E1_TIME, E2_TIME, e_window=0.5, minP=-1, maxP=-1, 
                   return_coverage=True, downsample=True):
    
    if minP<0:
        minP = np.abs(E1_TIME - E2_TIME)
    if maxP<0:
        maxP = (np.max(time) - np.min(time)) + minP

    P = np.arange(minP, maxP+e_window, e_window)
    E = np.arange(0, 1 + e_window/maxP, e_window/maxP)
    
    if return_coverage:
        PP, EE = np.meshgrid(P, E, indexing='ij')
        coverage = np.zeros_like(EE)

    pc = np.zeros_like(P)
    is1 = np.zeros_like(P)
    is2 = np.zeros_like(P)

    if downsample:
        hh, be = np.histogram(time, bins = np.arange(np.min(time), np.max(time), e_window/2))
        time = ((be[1:]+be[:-1])/2)[np.where((hh > 0))[0]]
        
    oki = np.where(((time < (E2_TIME - e_window)) |  (time > (E2_TIME + e_window))) & 
                   ((time < (E1_TIME - e_window)) | (time > (E1_TIME + e_window)))
                  )[0]

    for i in range(len(P)):
        win_i = e_window / P[i] # the eclipse window size in phase to examine at this period 

        if return_coverage:
            coverage[i,:-1], _ = np.histogram(((time[oki] - E1_TIME) % P[i]) / P[i], bins=E)
            
        pc_i = ((E2_TIME - E1_TIME) % P[i]) / P[i]
        phase_i = ((time[oki] - E1_TIME) % P[i]) / P[i]

        is1[i] = sum((phase_i <= win_i) | (phase_i >= (1-win_i)))
        is2[i] = sum((phase_i >= (pc_i - win_i)) & (phase_i <= (pc_i + win_i)))
    
    if return_coverage:
        return P, is1, is2, PP, EE, coverage
    else:
        return P, is1, is2 