import numpy as np
import scipy.signal
import librosa
from scipy.io import wavfile
# artificial impulse response, fading exponent multiplied by random variable
def generate_ir(duration=0.1, sample_rate=44100, decay_factor=60):
    t = np.linspace(0, duration, int(sample_rate * duration))
    ir = np.exp(-decay_factor * t)*np.random.uniform(-1,1,len(t))
    return ir

def add_reverb(data, ir):
    return scipy.signal.convolve(data, ir, mode='same') #convoultion in time domain 

#divide the mono into stereo sound, do math manipulation on both left and right channels
#use sine and cosine functions to create the 8d feeling, and by adding reverb
#the transition is much smoother.
#we add the reverb by convolve both channels with the impulse response.
#(another way to manage the convolution is to do a fourier transform of the impulse response
#and to the channels , and multiply them)

def create_8d_audio(signal_url, output_url, use_synthetic_ir=True, period_time=20):
    data, sample_rate = librosa.load(signal_url, sr=None, mono=True)
    data = data / np.max(np.abs(data))
    t = np.arange(len(data)) / sample_rate
    omega =  np.pi / period_time
    left_function = 0.1 + 0.8 * ((np.sin(omega * t))**2) 
    right_function = 0.1 + 0.8 *((np.cos(omega * t))**2)
    left_channel = data * left_function
    right_channel = data * right_function
    if use_synthetic_ir:
        ir = generate_ir()  
        left_channel = add_reverb(left_channel, ir)
        right_channel = add_reverb(right_channel, ir)
    max_val = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
    left_final = np.clip(left_channel / max_val * 32767, -32768, 32767).astype(np.int16)
    right_final = np.clip(right_channel / max_val * 32767, -32768, 32767).astype(np.int16)
    stereo_data = np.vstack((left_final, right_final)).T
    wavfile.write(output_url, sample_rate, stereo_data)
def main():
    input_url =r"input_path.wav"
    output_url =r"output_path.wav"
    create_8d_audio(input_url, output_url, use_synthetic_ir=True)

if __name__ == "__main__":
    main()