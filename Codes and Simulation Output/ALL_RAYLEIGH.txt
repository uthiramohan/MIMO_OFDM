%% Program to plot the BER of OFDM in Frequency selective channel

clc;
clear all;
close all;

N = 128;                                                % No of subcarriers
Ncp = 16;                                               % Cyclic prefix length
Ts = 1e-3;                                              % Sampling period of channel
Fd = 0;                                                 % Max Doppler frequency shift
Np = 4;                                                 % No of pilot symbols
M = 2;                                                  % No of symbols for PSK modulation
Nframes = 10^3;                                         % No of OFDM frames
D = round((M-1)*rand((N-2*Np),Nframes));
const = pskmod([0:M-1],M);
Dmod = pskmod(D,M);
Data = [zeros(Np,Nframes); Dmod ; zeros(Np,Nframes)];   % Pilot Insertion

%% OFDM symbol

IFFT_Data = (128/sqrt(120))*ifft(Data,N);
TxCy = [IFFT_Data((128-Ncp+1):128,:); IFFT_Data];       % Cyclic prefix
[r c] = size(TxCy);
Tx_Data = TxCy;

%% Frequency selective channel with 4 taps

tau = [0 1e-5 3.5e-5 12e-5];                            % Path delays
pdb = [0 -1 -1 -3];                                     % Avg path power gains
h = rayleighchan(Ts, Fd, tau, pdb);
h.StoreHistory = 0;
h.StorePathGains = 1;
h.ResetBeforeFiltering = 1;

%% SNR of channel

EbNo = 0:5:30;
EsNo= EbNo + 10*log10(120/128)+ 10*log10(128/144);      % symbol to noise ratio
snr= EsNo - 10*log10(128/144); 

%% Transmit through channel

berofdm1 = zeros(1,length(snr));
Rx_Data = zeros((N-2*Np),Nframes);
for i = 1:length(snr)
    for j = 1:c                                         % Transmit frame by frame
        hx = filter(h,Tx_Data(:,j).');                  % Pass through Rayleigh channel
        a = h.PathGains;
        AM = h.channelFilter.alphaMatrix;
        g = a*AM;                                       % Channel coefficients
        G(j,:) = fft(g,N);                              % DFT of channel coefficients
        y = awgn(hx,snr(i));                            % Add AWGN noise

%% Receiver
    
        Rx = y(Ncp+1:r);                                % Removal of cyclic prefix 
        FFT_Data = (sqrt(120)/128)*fft(Rx,N)./G(j,:);   % Frequency Domain Equalization
        Rx_Data(:,j) = pskdemod(FFT_Data(5:124),M);     % Removal of pilot and Demodulation 
    end
    berofdm1(i) = sum(sum(Rx_Data~=D))/((N-2*Np)*Nframes);
end

N = 128;                                                % No of subcarriers
Ncp = 16;                                               % Cyclic prefix length
Ts = 1e-3;                                              % Sampling period of channel
Fd = 0;                                                 % Max Doppler frequency shift
Np = 4;                                                 % No of pilot symbols
M = 4;                                                  % No of symbols for PSK modulation
Nframes = 10^3;                                         % No of OFDM frames
D = round((M-1)*rand((N-2*Np),Nframes));
const = pskmod([0:M-1],M);
Dmod = pskmod(D,M);
Data = [zeros(Np,Nframes); Dmod ; zeros(Np,Nframes)];   % Pilot Insertion

%% OFDM symbol

IFFT_Data = (128/sqrt(120))*ifft(Data,N);
TxCy = [IFFT_Data((128-Ncp+1):128,:); IFFT_Data];       % Cyclic prefix
[r c] = size(TxCy);
Tx_Data = TxCy;

%% Frequency selective channel with 4 taps

tau = [0 1e-5 3.5e-5 12e-5];                            % Path delays
pdb = [0 -1 -1 -3];                                     % Avg path power gains
h = rayleighchan(Ts, Fd, tau, pdb);
h.StoreHistory = 0;
h.StorePathGains = 1;
h.ResetBeforeFiltering = 1;

%% SNR of channel

EbNo = 0:5:30;
EsNo= EbNo + 10*log10(120/128)+ 10*log10(128/144);      % symbol to noise ratio
snr= EsNo - 10*log10(128/144); 

%% Transmit through channel

berofdm2 = zeros(1,length(snr));
Rx_Data = zeros((N-2*Np),Nframes);
for i = 1:length(snr)
    for j = 1:c                                         % Transmit frame by frame
        hx = filter(h,Tx_Data(:,j).');                  % Pass through Rayleigh channel
        a = h.PathGains;
        AM = h.channelFilter.alphaMatrix;
        g = a*AM;                                       % Channel coefficients
        G(j,:) = fft(g,N);                              % DFT of channel coefficients
        y = awgn(hx,snr(i));                            % Add AWGN noise

%% Receiver
    
        Rx = y(Ncp+1:r);                                % Removal of cyclic prefix 
        FFT_Data = (sqrt(120)/128)*fft(Rx,N)./G(j,:);   % Frequency Domain Equalization
        Rx_Data(:,j) = pskdemod(FFT_Data(5:124),M);     % Removal of pilot and Demodulation 
    end
    berofdm2(i) = sum(sum(Rx_Data~=D))/((N-2*Np)*Nframes);
end
%% Plot the BER

N = 128;                                                % No of subcarriers
Ncp = 16;                                               % Cyclic prefix length
Ts = 1e-3;                                              % Sampling period of channel
Fd = 0;                                                 % Max Doppler frequency shift
Np = 4;                                                 % No of pilot symbols
M = 8;                                                  % No of symbols for PSK modulation
Nframes = 10^3;                                         % No of OFDM frames
D = round((M-1)*rand((N-2*Np),Nframes));
const = pskmod([0:M-1],M);
Dmod = pskmod(D,M);
Data = [zeros(Np,Nframes); Dmod ; zeros(Np,Nframes)];   % Pilot Insertion

%% OFDM symbol

IFFT_Data = (128/sqrt(120))*ifft(Data,N);
TxCy = [IFFT_Data((128-Ncp+1):128,:); IFFT_Data];       % Cyclic prefix
[r c] = size(TxCy);
Tx_Data = TxCy;

%% Frequency selective channel with 4 taps

tau = [0 1e-5 3.5e-5 12e-5];                            % Path delays
pdb = [0 -1 -1 -3];                                     % Avg path power gains
h = rayleighchan(Ts, Fd, tau, pdb);
h.StoreHistory = 0;
h.StorePathGains = 1;
h.ResetBeforeFiltering = 1;

%% SNR of channel

EbNo = 0:5:30;
EsNo= EbNo + 10*log10(120/128)+ 10*log10(128/144);      % symbol to noise ratio
snr= EsNo - 10*log10(128/144); 

%% Transmit through channel

berofdm3 = zeros(1,length(snr));
Rx_Data = zeros((N-2*Np),Nframes);
for i = 1:length(snr)
    for j = 1:c                                         % Transmit frame by frame
        hx = filter(h,Tx_Data(:,j).');                  % Pass through Rayleigh channel
        a = h.PathGains;
        AM = h.channelFilter.alphaMatrix;
        g = a*AM;                                       % Channel coefficients
        G(j,:) = fft(g,N);                              % DFT of channel coefficients
        y = awgn(hx,snr(i));                            % Add AWGN noise

%% Receiver
    
        Rx = y(Ncp+1:r);                                % Removal of cyclic prefix 
        FFT_Data = (sqrt(120)/128)*fft(Rx,N)./G(j,:);   % Frequency Domain Equalization
        Rx_Data(:,j) = pskdemod(FFT_Data(5:124),M);     % Removal of pilot and Demodulation 
    end
    berofdm3(i) = sum(sum(Rx_Data~=D))/((N-2*Np)*Nframes);
end
%% Program to plot the BER of OFDM in Frequency selective channel

N = 128;                                                % No of subcarriers
Ncp = 16;                                               % Cyclic prefix length
Ts = 1e-3;                                              % Sampling period of channel
Fd = 0;                                                 % Max Doppler frequency shift
Np = 4;                                                 % No of pilot symbols
M = 2;                                                  % No of symbols for PSK modulation
Nframes = 10^3;                                         % No of OFDM frames
D = round((M-1)*rand((N-2*Np),Nframes));
const = dpskmod([0:M-1],M);
Dmod = dpskmod(D,M);
Data = [zeros(Np,Nframes); Dmod ; zeros(Np,Nframes)];   % Pilot Insertion

%% OFDM symbol

IFFT_Data = (128/sqrt(120))*ifft(Data,N);
TxCy = [IFFT_Data((128-Ncp+1):128,:); IFFT_Data];       % Cyclic prefix
[r c] = size(TxCy);
Tx_Data = TxCy;

%% Frequency selective channel with 4 taps

tau = [0 1e-5 3.5e-5 12e-5];                            % Path delays
pdb = [0 -1 -1 -3];                                     % Avg path power gains
h = rayleighchan(Ts, Fd, tau, pdb);
h.StoreHistory = 0;
h.StorePathGains = 1;
h.ResetBeforeFiltering = 1;

%% SNR of channel

EbNo = 0:5:30;
EsNo= EbNo + 10*log10(120/128)+ 10*log10(128/144);      % symbol to noise ratio
snr= EsNo - 10*log10(128/144); 

%% Transmit through channel

berofdm4 = zeros(1,length(snr));
Rx_Data = zeros((N-2*Np),Nframes);
for i = 1:length(snr)
    for j = 1:c                                         % Transmit frame by frame
        hx = filter(h,Tx_Data(:,j).');                  % Pass through Rayleigh channel
        a = h.PathGains;
        AM = h.channelFilter.alphaMatrix;
        g = a*AM;                                       % Channel coefficients
        G(j,:) = fft(g,N);                              % DFT of channel coefficients
        y = awgn(hx,snr(i));                            % Add AWGN noise

%% Receiver
    
        Rx = y(Ncp+1:r);                                % Removal of cyclic prefix 
        FFT_Data = (sqrt(120)/128)*fft(Rx,N)./G(j,:);   % Frequency Domain Equalization
        Rx_Data(:,j) = dpskdemod(FFT_Data(5:124),M);     % Removal of pilot and Demodulation 
    end
    berofdm4(i) = sum(sum(Rx_Data~=D))/((N-2*Np)*Nframes);
end

N = 128;                                                % No of subcarriers
Ncp = 16;                                               % Cyclic prefix length
Ts = 1e-3;                                              % Sampling period of channel
Fd = 0;                                                 % Max Doppler frequency shift
Np = 4;                                                 % No of pilot symbols
M = 4;                                                  % No of symbols for PSK modulation
Nframes = 10^3;                                         % No of OFDM frames
D = round((M-1)*rand((N-2*Np),Nframes));
const = dpskmod([0:M-1],M);
Dmod = dpskmod(D,M);
Data = [zeros(Np,Nframes); Dmod ; zeros(Np,Nframes)];   % Pilot Insertion

%% OFDM symbol

IFFT_Data = (128/sqrt(120))*ifft(Data,N);
TxCy = [IFFT_Data((128-Ncp+1):128,:); IFFT_Data];       % Cyclic prefix
[r c] = size(TxCy);
Tx_Data = TxCy;

%% Frequency selective channel with 4 taps

tau = [0 1e-5 3.5e-5 12e-5];                            % Path delays
pdb = [0 -1 -1 -3];                                     % Avg path power gains
h = rayleighchan(Ts, Fd, tau, pdb);
h.StoreHistory = 0;
h.StorePathGains = 1;
h.ResetBeforeFiltering = 1;

%% SNR of channel

EbNo = 0:5:30;
EsNo= EbNo + 10*log10(120/128)+ 10*log10(128/144);      % symbol to noise ratio
snr= EsNo - 10*log10(128/144); 

%% Transmit through channel

berofdm5 = zeros(1,length(snr));
Rx_Data = zeros((N-2*Np),Nframes);
for i = 1:length(snr)
    for j = 1:c                                         % Transmit frame by frame
        hx = filter(h,Tx_Data(:,j).');                  % Pass through Rayleigh channel
        a = h.PathGains;
        AM = h.channelFilter.alphaMatrix;
        g = a*AM;                                       % Channel coefficients
        G(j,:) = fft(g,N);                              % DFT of channel coefficients
        y = awgn(hx,snr(i));                            % Add AWGN noise

%% Receiver
    
        Rx = y(Ncp+1:r);                                % Removal of cyclic prefix 
        FFT_Data = (sqrt(120)/128)*fft(Rx,N)./G(j,:);   % Frequency Domain Equalization
        Rx_Data(:,j) = dpskdemod(FFT_Data(5:124),M);     % Removal of pilot and Demodulation 
    end
    berofdm5(i) = sum(sum(Rx_Data~=D))/((N-2*Np)*Nframes);
end
%% Plot the BER

N = 128;                                                % No of subcarriers
Ncp = 16;                                               % Cyclic prefix length
Ts = 1e-3;                                              % Sampling period of channel
Fd = 0;                                                 % Max Doppler frequency shift
Np = 4;                                                 % No of pilot symbols
M = 8;                                                  % No of symbols for PSK modulation
Nframes = 10^3;                                         % No of OFDM frames
D = round((M-1)*rand((N-2*Np),Nframes));
const = dpskmod([0:M-1],M);
Dmod = dpskmod(D,M);
Data = [zeros(Np,Nframes); Dmod ; zeros(Np,Nframes)];   % Pilot Insertion

%% OFDM symbol

IFFT_Data = (128/sqrt(120))*ifft(Data,N);
TxCy = [IFFT_Data((128-Ncp+1):128,:); IFFT_Data];       % Cyclic prefix
[r c] = size(TxCy);
Tx_Data = TxCy;

%% Frequency selective channel with 4 taps

tau = [0 1e-5 3.5e-5 12e-5];                            % Path delays
pdb = [0 -1 -1 -3];                                     % Avg path power gains
h = rayleighchan(Ts, Fd, tau, pdb);
h.StoreHistory = 0;
h.StorePathGains = 1;
h.ResetBeforeFiltering = 1;

%% SNR of channel

EbNo = 0:5:30;
EsNo= EbNo + 10*log10(120/128)+ 10*log10(128/144);      % symbol to noise ratio
snr= EsNo - 10*log10(128/144); 

%% Transmit through channel

berofdm6 = zeros(1,length(snr));
Rx_Data = zeros((N-2*Np),Nframes);
for i = 1:length(snr)
    for j = 1:c                                         % Transmit frame by frame
        hx = filter(h,Tx_Data(:,j).');                  % Pass through Rayleigh channel
        a = h.PathGains;
        AM = h.channelFilter.alphaMatrix;
        g = a*AM;                                       % Channel coefficients
        G(j,:) = fft(g,N);                              % DFT of channel coefficients
        y = awgn(hx,snr(i));                            % Add AWGN noise

%% Receiver
    
        Rx = y(Ncp+1:r);                                % Removal of cyclic prefix 
        FFT_Data = (sqrt(120)/128)*fft(Rx,N)./G(j,:);   % Frequency Domain Equalization
        Rx_Data(:,j) = dpskdemod(FFT_Data(5:124),M);     % Removal of pilot and Demodulation 
    end
    berofdm6(i) = sum(sum(Rx_Data~=D))/((N-2*Np)*Nframes);
end
N = 128;                                                % No of subcarriers
Ncp = 16;                                               % Cyclic prefix length
Ts = 1e-3;                                              % Sampling period of channel
Fd = 0;                                                 % Max Doppler frequency shift
Np = 4;                                                 % No of pilot symbols
M = 16;                                                  % No of symbols for PSK modulation
Nframes = 10^3;                                         % No of OFDM frames
D = round((M-1)*rand((N-2*Np),Nframes));
const = qammod([0:M-1],M);
Dmod = qammod(D,M);
Data = [zeros(Np,Nframes); Dmod ; zeros(Np,Nframes)];   % Pilot Insertion

%% OFDM symbol

IFFT_Data = (128/sqrt(120))*ifft(Data,N);
TxCy = [IFFT_Data((128-Ncp+1):128,:); IFFT_Data];       % Cyclic prefix
[r c] = size(TxCy);
Tx_Data = TxCy;

%% Frequency selective channel with 4 taps

tau = [0 1e-5 3.5e-5 12e-5];                            % Path delays
pdb = [0 -1 -1 -3];                                     % Avg path power gains
h = rayleighchan(Ts, Fd, tau, pdb);
h.StoreHistory = 0;
h.StorePathGains = 1;
h.ResetBeforeFiltering = 1;

%% SNR of channel

EbNo = 0:5:30;
EsNo= EbNo + 10*log10(120/128)+ 10*log10(128/144);      % symbol to noise ratio
snr= EsNo - 10*log10(128/144); 

%% Transmit through channel

berofdm7 = zeros(1,length(snr));
Rx_Data = zeros((N-2*Np),Nframes);
for i = 1:length(snr)
    for j = 1:c                                         % Transmit frame by frame
        hx = filter(h,Tx_Data(:,j).');                  % Pass through Rayleigh channel
        a = h.PathGains;
        AM = h.channelFilter.alphaMatrix;
        g = a*AM;                                       % Channel coefficients
        G(j,:) = fft(g,N);                              % DFT of channel coefficients
        y = awgn(hx,snr(i));                            % Add AWGN noise

%% Receiver
    
        Rx = y(Ncp+1:r);                                % Removal of cyclic prefix 
        FFT_Data = (sqrt(120)/128)*fft(Rx,N)./G(j,:);   % Frequency Domain Equalization
        Rx_Data(:,j) = qamdemod(FFT_Data(5:124),M);     % Removal of pilot and Demodulation 
    end
    berofdm7(i) = sum(sum(Rx_Data~=D))/((N-2*Np)*Nframes);
end

N = 128;                                                % No of subcarriers
Ncp = 16;                                               % Cyclic prefix length
Ts = 1e-3;                                              % Sampling period of channel
Fd = 0;                                                 % Max Doppler frequency shift
Np = 4;                                                 % No of pilot symbols
M = 64;                                                  % No of symbols for PSK modulation
Nframes = 10^3;                                         % No of OFDM frames
D = round((M-1)*rand((N-2*Np),Nframes));
const = qammod([0:M-1],M);
Dmod = qammod(D,M);
Data = [zeros(Np,Nframes); Dmod ; zeros(Np,Nframes)];   % Pilot Insertion

%% OFDM symbol

IFFT_Data = (128/sqrt(120))*ifft(Data,N);
TxCy = [IFFT_Data((128-Ncp+1):128,:); IFFT_Data];       % Cyclic prefix
[r c] = size(TxCy);
Tx_Data = TxCy;

%% Frequency selective channel with 4 taps

tau = [0 1e-5 3.5e-5 12e-5];                            % Path delays
pdb = [0 -1 -1 -3];                                     % Avg path power gains
h = rayleighchan(Ts, Fd, tau, pdb);
h.StoreHistory = 0;
h.StorePathGains = 1;
h.ResetBeforeFiltering = 1;

%% SNR of channel

EbNo = 0:5:30;
EsNo= EbNo + 10*log10(120/128)+ 10*log10(128/144);      % symbol to noise ratio
snr= EsNo - 10*log10(128/144); 

%% Transmit through channel

berofdm8 = zeros(1,length(snr));
Rx_Data = zeros((N-2*Np),Nframes);
for i = 1:length(snr)
    for j = 1:c                                         % Transmit frame by frame
        hx = filter(h,Tx_Data(:,j).');                  % Pass through Rayleigh channel
        a = h.PathGains;
        AM = h.channelFilter.alphaMatrix;
        g = a*AM;                                       % Channel coefficients
        G(j,:) = fft(g,N);                              % DFT of channel coefficients
        y = awgn(hx,snr(i));                            % Add AWGN noise

%% Receiver
    
        Rx = y(Ncp+1:r);                                % Removal of cyclic prefix 
        FFT_Data = (sqrt(120)/128)*fft(Rx,N)./G(j,:);   % Frequency Domain Equalization
        Rx_Data(:,j) = qamdemod(FFT_Data(5:124),M);     % Removal of pilot and Demodulation 
    end
    berofdm8(i) = sum(sum(Rx_Data~=D))/((N-2*Np)*Nframes);
end
%% Plot the BER

N = 128;                                                % No of subcarriers
Ncp = 16;                                               % Cyclic prefix length
Ts = 1e-3;                                              % Sampling period of channel
Fd = 0;                                                 % Max Doppler frequency shift
Np = 4;                                                 % No of pilot symbols
M = 128;                                                  % No of symbols for PSK modulation
Nframes = 10^3;                                         % No of OFDM frames
D = round((M-1)*rand((N-2*Np),Nframes));
const = qammod([0:M-1],M);
Dmod = qammod(D,M);
Data = [zeros(Np,Nframes); Dmod ; zeros(Np,Nframes)];   % Pilot Insertion

%% OFDM symbol

IFFT_Data = (128/sqrt(120))*ifft(Data,N);
TxCy = [IFFT_Data((128-Ncp+1):128,:); IFFT_Data];       % Cyclic prefix
[r c] = size(TxCy);
Tx_Data = TxCy;

%% Frequency selective channel with 4 taps

tau = [0 1e-5 3.5e-5 12e-5];                            % Path delays
pdb = [0 -1 -1 -3];                                     % Avg path power gains
h = rayleighchan(Ts, Fd, tau, pdb);
h.StoreHistory = 0;
h.StorePathGains = 1;
h.ResetBeforeFiltering = 1;

%% SNR of channel

EbNo = 0:5:30;
EsNo= EbNo + 10*log10(120/128)+ 10*log10(128/144);      % symbol to noise ratio
snr= EsNo - 10*log10(128/144); 

%% Transmit through channel

berofdm9 = zeros(1,length(snr));
Rx_Data = zeros((N-2*Np),Nframes);
for i = 1:length(snr)
    for j = 1:c                                         % Transmit frame by frame
        hx = filter(h,Tx_Data(:,j).');                  % Pass through Rayleigh channel
        a = h.PathGains;
        AM = h.channelFilter.alphaMatrix;
        g = a*AM;                                       % Channel coefficients
        G(j,:) = fft(g,N);                              % DFT of channel coefficients
        y = awgn(hx,snr(i));                            % Add AWGN noise

%% Receiver
    
        Rx = y(Ncp+1:r);                                % Removal of cyclic prefix 
        FFT_Data = (sqrt(120)/128)*fft(Rx,N)./G(j,:);   % Frequency Domain Equalization
        Rx_Data(:,j) = qamdemod(FFT_Data(5:124),M);     % Removal of pilot and Demodulation 
    end
    berofdm9(i) = sum(sum(Rx_Data~=D))/((N-2*Np)*Nframes);
end
figure;
semilogy(EbNo,berofdm1,'--+c');
hold on
semilogy(EbNo,berofdm2,'--om');
hold on
semilogy(EbNo,berofdm3,'--*y');
hold on
semilogy(EbNo,berofdm4,'--xr');
hold on
semilogy(EbNo,berofdm5,'--sg');
hold on
semilogy(EbNo,berofdm6,'--db');
hold on
semilogy(EbNo,berofdm7,'--^c');
hold on
semilogy(EbNo,berofdm8,'-->k');
hold on
semilogy(EbNo,berofdm9,'--<b');
grid on;
legend('BPSK','QPSK','8PSK','DBPSK','DQPSK','8DPSK','16-QAM','64-QAM','256-QAM','Location','Southwest');
title('SNR vs BER for BPSK/QPSK/8-PSK/DBPSK/DQPSK/8-DPSK/16,64,256-QAM MIMO OFDM over Rayleigh Fading Channel');
xlabel('EbNo');
ylabel('BER');