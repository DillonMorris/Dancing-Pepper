import pyaudio
import wave
import time
import struct
import numpy as np
import matplotlib as plt
import pygame
import math
import matplotlib.pyplot as pipt
from scipy import signal  

audio = pyaudio.PyAudio()

start_time = time.time()


FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 0.025
WAVE_OUTPUT_FILENAME = "file.wav"
fouriers_per_second = 24  # Frames per second
fourier_spread = 1.0 / fouriers_per_second
fourier_width = fourier_spread

def new_stream():
	stream = audio.open(format=FORMAT,
		channels=CHANNELS - 1,
		rate=RATE,
		input=True,
		frames_per_buffer=CHUNK)
	return  stream


def getFFT(data,rate):
    """Given some data and rate, returns FFTfreq and FFT (half)."""
    data=data*np.hamming(len(data))
    fft=np.fft.fftshift(data)
    fft=np.abs(fft)
    #fft=10*np.log10(fft)
    freq=np.fft.fftfreq(len(fft),1.0/rate)
    return freq[:int(len(freq)/2)],fft[:int(len(fft)/2)]





def getBandWidth():
	return (2.0 / RATE) * (RATE / 2.0)



def freqToIndex(f):
# If f (frequency is lower than the bandwidth of spectrum[0]
	if f < getBandWidth() / 2: 
		return 0 
	if f > (RATE / 2) - (getBandWidth() / 2): 
		return (RATE - 1)
	fraction = float(f) / float(RATE)
	index = round(RECORD_SECONDS * RATE * fraction)
	return index



fft_averages = []


def average_fft_bands(fft_array):
	print(average_fft_bands)
	num_bands = 20  # The number of frequency bands (12 = 1 octave)
	del fft_averages[:]
	
	for band in range(0, num_bands):
		avg = 0.0

		if band == 0:
			lowFreq = int(0)
		else:
			lowFreq = int(int(RATE / 2) / float(2 ** (num_bands - band)))
		hiFreq = int((RATE / 2) / float(2 ** ((num_bands - 1) - band)))
		lowBound = int(freqToIndex(lowFreq))
		hiBound = int(freqToIndex(hiFreq))
		if(hiBound >= 1023):
			hiBound = 1022
		for j in range(lowBound, hiBound):

			avg += fft_array[j]
		avg /= (hiBound - lowBound + 1)
		fft_averages.append(avg)




fourier_width_index = fourier_width * float(44100)

audio = pyaudio.PyAudio()

# start Recording

#stream1 = new_stream()

#sin_data = []

print("recording...")
frames = []
timer = 0
while True:

#	print("new stream")
#	print("timer ", timer)
	time_now = (time.time() - start_time)
#	print("",time_now )
	n_s = new_stream()

	data = 0
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):

#		print("i= ", i)

		data = n_s.read(CHUNK)
		#frames.append(data)
		data_int = struct.unpack(str(2 * CHUNK) + 'B', data)
		data_int = np.array(struct.unpack(str(2 * CHUNK) + 'B', data), dtype='b')[::2] + 127

	data_int = signal.detrend(data_int)	
	print(len(data_int))

	freakyboy , figetdude = getFFT(data_int, 32768)

	fidgetbroskendomanthefith = figetdude/1000

	pipt.plot(freakyboy, fidgetbroskendomanthefith)
	
#	print('dil')
	print(len(fidgetbroskendomanthefith))
	print(len(freakyboy))

	av_freak_holder = [0,0,0,0]

	perrie = len(fidgetbroskendomanthefith)

	av1 = 0
	for i in range(0,(perrie/4)-1):
		av1 += fidgetbroskendomanthefith[i]
	av2 = 0
	for i in range(perrie/4,(perrie/2)-1):
		av2 += fidgetbroskendomanthefith[i]
	av3 = 0 
	for i in range(perrie/2,((3*perrie)/4)-1):
		av3 += fidgetbroskendomanthefith[i]
	av4 = 0
	for i in range((3*perrie/4), perrie-1):
		av4 += fidgetbroskendomanthefith[i]

	
	av_freak_holder[0] = av1/(perrie/4)
	av_freak_holder[1] = av2/(perrie/4)
	av_freak_holder[2] = av3/(perrie/4)
	av_freak_holder[3] = av4/(perrie/4)
	total_amplitude = (av_freak_holder[0]+av_freak_holder[1]+av_freak_holder[2]+av_freak_holder[3])


	
	print(total_amplitude)

	n_s.stop_stream()

	n_s.close()
#	pipt.show()
	"""

	#	print(data_int)
	#	print(len(data_int))
#		fft_data = abs(np.fft.fft(data_int))/10000

		
		
		for i in range(0,1024):
			dil = np.sin(i)
			#print(dil)
			sin_data.append( dil)
			#print(sin_data)

		#print(sin_data)
		
		fft_data = np.fft.fft(sin_data)
		
		spec_y = [np.sqrt(c.real * 2 + c.imag * 2) for c in fft_data]

		#print(spec_y)
		#print(len(spec_y))
		#fft_data = ((2 ** .5) / RECORD_SECONDS * RATE)
#		xarray = list(range(1024))
		xarray = np.fft.fftfreq(CHUNK, d=1.0/RATE)
		
		for i in range (0,len(xarray)):
			print(xarray[i])

		pipt.plot(xarray, spec_y)
		##
		#print('fft_averages')
		#print('dil')
		#print(average_fft_bands(fft_data))

		#print('dil')
		#y_axis = fft_averages
		#pipt.show()
		sin_data = []
"""
"""		
		for j in range(8, 
			COLOR = GREEN
			y_axis[j] = y_axis[j] / 10000000

			if (max(y_axis) > 20000000000):
				COLOR = RED
			else:
				COLOR = GREEN

				if (y_axis[j] > 750 ):
					pygame.draw.circle(windowSurface, COLOR, (int(150 + (y_axis[j])/5), 30*j ), 15, 5)

					timer += 1
					pygame.display.update()
					windowSurface.fill(BLACK)
					n_s.stop_stream()

					n_s.close()

					in_range_y = []
					print("finished recording")
					print(y_axis)

"""
