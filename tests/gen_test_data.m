clearvars;

FS = 48000;

L = 4096;
f0 = 1000;

x = zeros(L,1);
t = (0:L-1)' / FS;

partial_count = 20;
for n = 1:2:partial_count
    x = x + (1/n) * sin(2 * pi * f0*n * t);
end

x = x / max(abs(x));
x = x * 0.8;

xf = fft(x);

txt_friendly_spectrum = [real(xf) imag(xf)];
writematrix(txt_friendly_spectrum, "test_signal_spectrum_4096.txt");

xf_oversampled = fft(x, L*2);
txt_friendly_spectrum = [real(xf_oversampled) imag(xf_oversampled)];
writematrix(txt_friendly_spectrum, "test_signal_spectrum_8192.txt");

mag_spectrum = abs(xf);
writematrix(mag_spectrum, "test_signal_mag_spectrum.txt");

db_spectrum = 20*log10(mag_spectrum);
writematrix(db_spectrum, "test_signal_db_spectrum.txt");

audiowrite("test_signal.wav", x, FS, BitsPerSample=32)

figure(1);
subplot(211);
plot(x);

subplot(212);


f = FS/L*(0:L-1);

plot(f, abs(xf));

cepstrum = rceps(x);
figure(2);
plot(cepstrum);

writematrix(cepstrum, "test_signal_cepstrum.txt");


[autocorr, lags] = xcorr(x, 'normalized');
autocorr = autocorr(find(lags==0):end);
writematrix(autocorr, "test_signal_autocorr.txt");

[s,f,t] = stft(x, FS, Window=hann(128), OverlapLength=127, FFTLength=256, FrequencyRange="onesided");

figure(3);
sdb = (abs(s));
imagesc(sdb);

writematrix(sdb, "test_signal_spectrogram.txt");