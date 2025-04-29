import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def generate_sine_wave(frequency, duration=1.0, sample_rate=22050, amplitude=1.0):
    """
    Generate a sine wave with specified parameters
    
    Args:
        frequency (float): Frequency of the sine wave in Hz
        duration (float): Duration of the audio in seconds
        sample_rate (int): Number of samples per second
        amplitude (float): Amplitude of the sine wave
    
    Returns:
        tuple: (audio_data, sample_rate)
    """
    # Generate time points
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate sine wave
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    
    return sine_wave, sample_rate

def plot_sine_wave(audio_data, sample_rate, frequency, output_path='sine_wave_plot.png'):
    """Plot the sine wave in time and frequency domain"""
    duration = len(audio_data) / sample_rate
    t = np.linspace(0, duration, len(audio_data), False)
    
    plt.figure(figsize=(15, 10))
    
    # Time domain plot
    plt.subplot(3, 1, 1)
    plt.plot(t, audio_data)
    plt.title(f'Sine Wave - {frequency} Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # FFT
    fft = np.fft.rfft(audio_data)
    freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
    
    # Frequency domain plot (linear)
    plt.subplot(3, 1, 2)
    plt.plot(freqs, np.abs(fft))
    plt.title('Frequency Spectrum (Linear)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    # Frequency domain plot (log scale)
    plt.subplot(3, 1, 3)
    plt.semilogy(freqs, np.abs(fft))
    plt.title('Frequency Spectrum (Log Scale)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (log)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    # Parameters
    frequency = 440  # Hz (A4 note)
    duration = 1.0   # seconds
    sample_rate = 22050  # Hz
    amplitude = 0.5  # Amplitude (between 0 and 1)
    
    # Generate sine wave
    audio_data, sample_rate = generate_sine_wave(
        frequency=frequency,
        duration=duration,
        sample_rate=sample_rate,
        amplitude=amplitude
    )
    
    # Save audio file
    output_filename = f'sine_wave_{frequency}hz.wav'
    sf.write(output_filename, audio_data, sample_rate)
    print(f"Generated sine wave saved to {output_filename}")
    
    # Plot the wave
    plot_sine_wave(audio_data, sample_rate, frequency)
    print("Plot saved to sine_wave_plot.png")
