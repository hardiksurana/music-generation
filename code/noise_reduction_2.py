import numpy as np
import librosa
import argparse
import time


def main():
    parser = argparse.ArgumentParser(
        prog = 'The Noise Reduction (Spectral Subtraction)',
        usage = 'シンプルなスペクトルサブトラクションで, ノイズを軽減します.',
        description = 'python3 NoiseReduction.py -i [Input Filename] -o [Output Filename] -s [Noise start time(sec)] -f [Noise finish time(sec)]',
        epilog = 'MIT Licensce',
        add_help=True       
    )
    parser.add_argument('-i', '--input', help = 'Input Filename', required = True)
    parser.add_argument('-o', '--output', help = 'Output Filename (wav)', required = True)
    parser.add_argument('-s', '--start', help = 'Cut Noise Sound (Start Time [sec])', required = True)
    parser.add_argument('-f', '--finish', help = 'Cut Noise Sound (Finish Time [sec])', required = True)
    args = parser.parse_args()

    data, sr = librosa.load(args.input)

    # short-time Fourier transfor (STFT)
    #  (n_fft = 2048, hop_length = win_length(=n_fft) / 4, window = 'hann')
    # D: np.ndarray [shape=(1+n_fft / 2, t) T = t * hop_length])
    S = np.abs(librosa.stft(data))

    # Convert a power spectrogram to decibel(dB)
    D = librosa.power_to_db(S**2)

    # Calc Noise FrameRate
    _n_fft = 2048
    _hop_length = _n_fft / 4
    noise_start = int(_hop_length * float(args.start))
    noise_finish = int(_hop_length * float(args.finish))

    # Noise Copy and calc Average powers
    noise_D = D[:, noise_start : noise_finish]
    noise_Ave = np.average(noise_D, axis = 1)

    # Calc Spectral Subtraction
    D = D.transpose()
    SS = D - noise_Ave
    SS = SS.transpose()

    # Convert decibel to power spectrogram
    SSP = librosa.db_to_power(SS)
    
    # Inverse short-time Fourier transfor(ISTFT)
    OutputS = librosa.istft(SSP)

    # 正規化(normalize)
    OutputS = librosa.util.normalize(OutputS)

    # Output File (WAV)
    librosa.output.write_wav(args.output, OutputS, sr)    

if __name__ == '__main__':
    main()
