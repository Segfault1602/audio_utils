clear all;
clf;

fs = 48000;
ts = 0:1/fs:(1-1/fs);
x = chirp(ts, 100, 0.5, 5000, 'quadratic');
audiowrite("chirp.wav", x, fs);

nfft = 1024;
overlap = 512;
w = hann(nfft);

x_buffer = buffer(x, nfft, overlap, "nodelay");

spec_data = zeros(nfft/2 + 1, size(x_buffer, 2));

for n = 1:size(x_buffer,2)
    xf = abs(fft(x_buffer(:,n).*w));
    spec_data(:,n) = xf(1:nfft/2 + 1);
end

spec_data = spec_data(:,1:end-1);

imagesc(pow2db(spec_data))
axis xy
xlabel("Time (s)")
ylabel("Frequency (Hz)")
colorbar

test_data = readmatrix("../spectrogram.txt")';

figure(2)
imagesc(pow2db(test_data));
axis xy
xlabel("Time (s)")
ylabel("Frequency (Hz)")
colorbar
