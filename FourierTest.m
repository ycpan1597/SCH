% Purpose: to test if slicing a signal changes its frequency spectrum and
% learn proper implementation of FFT on Matlab

close all; clear all; clc
Fs = 100;                % Sampling Freq (in Hz)
T = 1/Fs;               % Sampling Period (in sec)
L = 5;                  % Length of Signal (in sec)
nfft = L * Fs;          % Number of Points
t = 0:T:L;              % Time Vector
freq = 3;               % Signal Freq (in Hz)


x = sin(freq*2*pi*t);   
X = fft(x, nfft);
X_half = X(1:nfft/2);    % Spectrum of real signal is always symmetric, so we throw away half of it
f = (0:nfft/2 - 1) * Fs/nfft;
figure()
stem(f, abs(X_half))


x_broken = [x(1:17), x(50:80)];
nfft_broken = length(x_broken); 
X_broken = fft(x_broken, nfft_broken);
X_broken_half = X_broken(1:nfft_broken/2);
f_broken = (0:nfft_broken/2 - 1) * Fs/nfft_broken;
figure()
stem(f_broken, abs(X_broken_half))

% brokenSignal = [x(1:17), x(50:80)];
% freqAxis = [-pi:2*pi/length(t):pi];
% 
% figure()
% subplot(2,1,1)
% plot(t, x)
% subplot(2,1,2)
% spec = fft(x);
% plot(abs(spec))
% 
% 
% figure()
% subplot(2,1,1)
% plot([t(1:17), t(50:80)], brokenSignal)
% subplot(2,1,2)
% brokenSpec = fft(brokenSignal);
% plot(abs(brokenSpec))

% Conclusion: slicing significantly impacts the frequency spectrum.. 
% Discussion: Is it okay to slice the signal first? 
