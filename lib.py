import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert


def plotsEEGFeatures(data,words,sr,s=5,duration=5,low=12,high=30,winL=0.2):
    ### Visualizing raw data
    ### Here, we inspect the raw data and the corresponding labels
    ### and see if certain frequency ranges contain obvious information
    ### or artifacts

    # Theta = 4-7 Hz
    # Alpha = 8-12
    # Beta = 12-30
    # Gamma = 30-50
    # High-Gamma = 60-90
    lowFrequencyCutoff=low
    highFrequencyCutoff=high
    #Start and end (in seconds) of the eeg bit
    start=s
    end=s+duration
    x=np.arange(start*sr,end*sr)/sr-start
    #Creating a butterworth filter
    nyquist = sr/2
    b, a = butter(3, [lowFrequencyCutoff/nyquist,highFrequencyCutoff/nyquist], btype='bp')
    # Cutting out the corresponding data

    dat = data[int(sr*start):int(sr*end)]
    # Initializing the plot
    fig, ax = plt.subplots(5,figsize=[20,8],sharex=True)
    # Plotting the words binarized (1 if a word is spoken, 0 else)
    ax[0].set_title('Label')
    lbl = words[int(sr*start):int(sr*end)]!=''
    ax[0].plot(x,lbl,label='Words')
    # Plotting the raw data
    ax[1].set_title('Raw')
    ax[1].plot(x,dat,label='Raw')
    # Bandpass filtering and plotting
    filtered=filtfilt(b,a,dat)
    ax[2].set_title('Band-passed')
    ax[2].plot(x,filtered,label='Band-passed')
    # Calculating the hilbert envelope of the filtered signal
    hilEnv=np.abs(hilbert(filtered))
    ax[3].set_title('Envelope')
    ax[3].plot(x,hilEnv,label='Band-passed')
    # Windowing
    winLength = winL
    for i in np.arange(np.min(x),np.max(x),winLength):
        ax[3].axvline(x=i,color='r')
    winEnv=hilEnv
    for win in range(0,len(hilEnv),int(winLength*sr)):
        winEnv[win:win+int(winLength*sr)]=np.mean(hilEnv[win:win+int(winLength*sr)])
    ax[4].set_title('Windowed Envelope')
    ax[4].plot(x,winEnv,label='Windowed Envelope')
    ax[4].set_xlabel('Time in seconds')

    #Making it a bit prettier
    for axs in ax:
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.spines['bottom'].set_visible(False)
    plt.show()
    
def plotsEEG(data,sr,s=5,duration=5,low=12,high=30):
    low=int(low)
    high=int(high)
    e=s+duration
    x=np.arange(s*sr,sr*e)/sr-s
    b, a = butter(3, [low/(sr/2),high/(sr/2)], btype='bp')
    fig, ax = plt.subplots(3,figsize=[20,4],sharex=True)
    dat = data[sr*s:sr*e]
    ax[0].set_title('Raw')
    ax[0].plot(x,dat,label='Raw')
    filtered=filtfilt(b,a,dat)
    ax[1].set_title('Band-passed')
    ax[1].plot(x,filtered,label='Band-passed')
    hilEnv=np.abs(hilbert(filtered))
    ax[2].set_title('Envelope')
    ax[2].plot(x,hilEnv,label='Band-passed')

    for axs in ax:
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.spines['bottom'].set_visible(False)
    plt.show()
    
def plotsEEGSpec(data,sr,s=5,duration=5,winLength=0.1):
    e=s+duration
    x=np.arange(s*sr,sr*e)/sr-s
    fig, ax = plt.subplots(2,figsize=[20,4])
    noise = np.sin(2*np.pi*x*60)*30
    dat = data[sr*s:sr*e]+noise
    
    ax[0].plot(x,dat)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    
    dat = dat[:int(winLength*sr)]
    freqs = np.fft.rfftfreq(dat.shape[0], d=1/sr)
    freqPower=np.log(np.abs(np.fft.rfft(dat)))

    ax[1].plot(freqs,freqPower)
    ax[1].set_xlabel('Frequency')
    ax[1].set_ylabel('Power')
    
    
    ax[0].axvline(x=winLength,color='r')

    plt.show()
    
def plotRawSpec(data,sr,s=5,duration=5,winLength=0.05):
    e=s+duration
    x=np.arange(s*sr,sr*e)/sr-s
    fig, ax = plt.subplots(2,figsize=[20,4])
    dat = data[sr*s:sr*e]
    ax[0].set_title('Raw')
    ax[0].plot(x,dat,label='Raw')
   
    for i in np.arange(np.min(x),np.max(x),winLength):
        ax[0].axvline(x=i,color='r')
    ax[0].set_xlim(0,x[-1])
    numWindows=int(np.floor((dat.shape[0])/(winLength*sr)))
    numSpecs=int(np.floor(winLength*sr / 2 + 1))
    spec=np.zeros((numWindows,numSpecs))
    freqs = np.fft.rfftfreq(int(winLength*sr), d=1/sr)
    #print(freqs)
    for w in range(numWindows):
        s=int(w*winLength*sr)
        e=int(s+winLength*sr)
        spec[w,:]=np.abs(np.fft.rfft(dat[s:e]))
    spec=np.array(spec)
   
    ax[1].set_title('Spectrogram')
    spec=np.log(spec)
    ax[1].imshow(np.flipud(spec.T), aspect='auto', cmap='viridis')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel(str(len(freqs)) + ' Frequs')
    ax[1].set_yticks([0,spec.shape[1]])
    ax[1].set_yticklabels([str(sr/2),str(0),])
    for axs in ax:
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.spines['bottom'].set_visible(False)
    plt.show()
    