function compute_symbol_error_rate_qam_ofdm_awgn()


    M = 2;                 % Modulation alphabet
    k = log2(M);           % Bits/symbol
    numSC = 128;           % Number of OFDM subcarriers
    cpLen = 32;            % OFDM cyclic prefix length
    maxBitErrors = 100;    % Maximum number of bit errors
    maxNumBits = 1e7;      % Maximum number of bits transmitted

  
    hQPSKMod = comm.DBPSKModulator;
    hQPSKDemod = comm.DBPSKDemodulator;
    
    hOFDMmod = comm.OFDMModulator('FFTLength',numSC,'CyclicPrefixLength',cpLen);
    hOFDMdemod = comm.OFDMDemodulator('FFTLength',numSC,'CyclicPrefixLength',cpLen);
   
    hChan = comm.AWGNChannel('NoiseMethod','Variance', ...
        'VarianceSource','Input port');
  
    hError = comm.ErrorRate('ResetInputPort',true);

   
    ofdmInfo = info(hOFDMmod)
   
    numDC = ofdmInfo.DataInputSize(1)
   
    frameSize = [k*numDC 1];

   
    EbNoVec = (0:33)';
    snrVec = EbNoVec + 10*log10(k) + 10*log10(numDC/numSC);
    
    berVec = zeros(length(EbNoVec),3);
    errorStats = zeros(1,3);
       for m = 1:length(EbNoVec)
        snr = snrVec(m);

        while errorStats(2) <= maxBitErrors && errorStats(3) <= maxNumBits
            dataIn = randi([0,1],frameSize);              % Generate binary data
            qpskTx = step(hQPSKMod,dataIn);               % Apply QPSK modulation
            txSig = step(hOFDMmod,qpskTx);                % Apply OFDM modulation
            powerDB = 10*log10(var(txSig));               % Calculate Tx signal power
            noiseVar = 10.^(0.1*(powerDB-snr));           % Calculate the noise variance
            rxSig = step(hChan,txSig,noiseVar);           % Pass the signal through a noisy channel
            qpskRx = step(hOFDMdemod,rxSig);              % Apply OFDM demodulation
            dataOut = step(hQPSKDemod,qpskRx);            % Apply QPSK demodulation
            errorStats = step(hError,dataIn,dataOut,0);   % Collect error statistics
        end

        berVec(m,:) = errorStats;                         % Save BER data
        errorStats = step(hError,dataIn,dataOut,1);       % Reset the error rate calculator
    end

    
hQPSKMod = comm.BPSKModulator;
hQPSKDemod = comm.BPSKDemodulator;

hOFDMmod = comm.OFDMModulator('FFTLength',numSC,'CyclicPrefixLength',cpLen);
hOFDMdemod = comm.OFDMDemodulator('FFTLength',numSC,'CyclicPrefixLength',cpLen);

hChan = comm.AWGNChannel('NoiseMethod','Variance', ...
    'VarianceSource','Input port');

hError = comm.ErrorRate('ResetInputPort',true);


ofdmInfo = info(hOFDMmod)

numDC = ofdmInfo.DataInputSize(1)

frameSize = [k*numDC 1];


EbNoVec = (0:33)';
snrVec = EbNoVec + 10*log10(k) + 10*log10(numDC/numSC);

berVec1 = zeros(length(EbNoVec),3);
errorStats = zeros(1,3);

for m = 1:length(EbNoVec)
    snr = snrVec(m);
    
    while errorStats(2) <= maxBitErrors && errorStats(3) <= maxNumBits
        dataIn = randi([0,1],frameSize);              % Generate binary data
        qpskTx = step(hQPSKMod,dataIn);               % Apply QPSK modulation
        txSig = step(hOFDMmod,qpskTx);                % Apply OFDM modulation
        powerDB = 10*log10(var(txSig));               % Calculate Tx signal power
        noiseVar = 10.^(0.1*(powerDB-snr));           % Calculate the noise variance
        rxSig = step(hChan,txSig,noiseVar);           % Pass the signal through a noisy channel
        qpskRx = step(hOFDMdemod,rxSig);              % Apply OFDM demodulation
        dataOut = step(hQPSKDemod,qpskRx);            % Apply QPSK demodulation
        errorStats = step(hError,dataIn,dataOut,0);   % Collect error statistics
    end
    
    berVec1(m,:) = errorStats;                         % Save BER data
    errorStats = step(hError,dataIn,dataOut,1);       % Reset the error rate calculator
end


    M = 4;                 % Modulation alphabet
    k = log2(M);           % Bits/symbol
    numSC = 128;           % Number of OFDM subcarriers
    cpLen = 32;            % OFDM cyclic prefix length
    maxBitErrors = 100;    % Maximum number of bit errors
    maxNumBits = 1e7;      % Maximum number of bits transmitted
hQPSKMod = comm.QPSKModulator('BitInput',true);
hQPSKDemod = comm.QPSKDemodulator('BitOutput',true);

hOFDMmod = comm.OFDMModulator('FFTLength',numSC,'CyclicPrefixLength',cpLen);
hOFDMdemod = comm.OFDMDemodulator('FFTLength',numSC,'CyclicPrefixLength',cpLen);

hChan = comm.AWGNChannel('NoiseMethod','Variance', ...
    'VarianceSource','Input port');

hError = comm.ErrorRate('ResetInputPort',true);


ofdmInfo = info(hOFDMmod)

numDC = ofdmInfo.DataInputSize(1)

frameSize = [k*numDC 1];


EbNoVec = (0:33)';
snrVec = EbNoVec + 10*log10(k) + 10*log10(numDC/numSC);

berVec3 = zeros(length(EbNoVec),3);
errorStats = zeros(1,3);

for m = 1:length(EbNoVec)
    snr = snrVec(m);
    
    while errorStats(2) <= maxBitErrors && errorStats(3) <= maxNumBits
        dataIn = randi([0,1],frameSize);              % Generate binary data
        qpskTx = step(hQPSKMod,dataIn);               % Apply QPSK modulation
        txSig = step(hOFDMmod,qpskTx);                % Apply OFDM modulation
        powerDB = 10*log10(var(txSig));               % Calculate Tx signal power
        noiseVar = 10.^(0.1*(powerDB-snr));           % Calculate the noise variance
        rxSig = step(hChan,txSig,noiseVar);           % Pass the signal through a noisy channel
        qpskRx = step(hOFDMdemod,rxSig);              % Apply OFDM demodulation
        dataOut = step(hQPSKDemod,qpskRx);            % Apply QPSK demodulation
        errorStats = step(hError,dataIn,dataOut,0);   % Collect error statistics
    end
    
    berVec3(m,:) = errorStats;                         % Save BER data
    errorStats = step(hError,dataIn,dataOut,1);       % Reset the error rate calculator
end


    hQPSKMod = comm.DQPSKModulator('BitInput',true);
    hQPSKDemod = comm.DQPSKDemodulator('BitOutput',true);

    hOFDMmod = comm.OFDMModulator('FFTLength',numSC,'CyclicPrefixLength',cpLen);
    hOFDMdemod = comm.OFDMDemodulator('FFTLength',numSC,'CyclicPrefixLength',cpLen);

    hChan = comm.AWGNChannel('NoiseMethod','Variance', ...
        'VarianceSource','Input port');

    hError = comm.ErrorRate('ResetInputPort',true);


    ofdmInfo = info(hOFDMmod)

    numDC = ofdmInfo.DataInputSize(1)

    frameSize = [k*numDC 1];

    EbNoVec = (0:33)';
    snrVec5 = EbNoVec + 10*log10(k) + 10*log10(numDC/numSC);

    berVec5 = zeros(length(EbNoVec),3);
    errorStats = zeros(1,3);
   
    for m = 1:length(EbNoVec)
        snr = snrVec5(m);

        while errorStats(2) <= maxBitErrors && errorStats(3) <= maxNumBits
            dataIn = randi([0,1],frameSize);              % Generate binary data
            qpskTx = step(hQPSKMod,dataIn);               % Apply QPSK modulation
            txSig = step(hOFDMmod,qpskTx);                % Apply OFDM modulation
            powerDB = 10*log10(var(txSig));               % Calculate Tx signal power
            noiseVar = 10.^(0.1*(powerDB-snr));           % Calculate the noise variance
            rxSig = step(hChan,txSig,noiseVar);           % Pass the signal through a noisy channel
            qpskRx = step(hOFDMdemod,rxSig);              % Apply OFDM demodulation
            dataOut = step(hQPSKDemod,qpskRx);            % Apply QPSK demodulation
            errorStats = step(hError,dataIn,dataOut,0);   % Collect error statistics
        end

        berVec5(m,:) = errorStats;                         % Save BER data
        errorStats = step(hError,dataIn,dataOut,1);       % Reset the error rate calculator
    end

    
    
    M = 8;                 % Modulation alphabet
k = log2(M);           % Bits/symbol
numSC = 128;           % Number of OFDM subcarriers
cpLen = 32;            % OFDM cyclic prefix length
maxBitErrors = 100;    % Maximum number of bit errors
maxNumBits = 1e7;      % Maximum number of bits transmitted


hQPSKMod = comm.PSKModulator('BitInput',true);
hQPSKDemod = comm.PSKDemodulator('BitOutput',true);

hOFDMmod = comm.OFDMModulator('FFTLength',numSC,'CyclicPrefixLength',cpLen);
hOFDMdemod = comm.OFDMDemodulator('FFTLength',numSC,'CyclicPrefixLength',cpLen);

hChan = comm.AWGNChannel('NoiseMethod','Variance', ...
    'VarianceSource','Input port');

hError = comm.ErrorRate('ResetInputPort',true);


ofdmInfo = info(hOFDMmod)

numDC = ofdmInfo.DataInputSize(1)

frameSize = [k*numDC 1];


EbNoVec = (0:33)';
snrVec = EbNoVec + 10*log10(k) + 10*log10(numDC/numSC);

berVec4 = zeros(length(EbNoVec),3);
errorStats = zeros(1,3);

for m = 1:length(EbNoVec)
    snr = snrVec(m);
    
    while errorStats(2) <= maxBitErrors && errorStats(3) <= maxNumBits
        dataIn = randi([0,1],frameSize);              % Generate binary data
        qpskTx = step(hQPSKMod,dataIn);               % Apply QPSK modulation
        txSig = step(hOFDMmod,qpskTx);                % Apply OFDM modulation
        powerDB = 10*log10(var(txSig));               % Calculate Tx signal power
        noiseVar = 10.^(0.1*(powerDB-snr));           % Calculate the noise variance
        rxSig = step(hChan,txSig,noiseVar);           % Pass the signal through a noisy channel
        qpskRx = step(hOFDMdemod,rxSig);              % Apply OFDM demodulation
        dataOut = step(hQPSKDemod,qpskRx);            % Apply QPSK demodulation
        errorStats = step(hError,dataIn,dataOut,0);   % Collect error statistics
    end
    
    berVec4(m,:) = errorStats;                         % Save BER data
    errorStats = step(hError,dataIn,dataOut,1);       % Reset the error rate calculator
end

    hQPSKMod = comm.DPSKModulator('BitInput',true);
    hQPSKDemod = comm.DPSKDemodulator('BitOutput',true);

    hOFDMmod = comm.OFDMModulator('FFTLength',numSC,'CyclicPrefixLength',cpLen);
    hOFDMdemod = comm.OFDMDemodulator('FFTLength',numSC,'CyclicPrefixLength',cpLen);

    hChan = comm.AWGNChannel('NoiseMethod','Variance', ...
        'VarianceSource','Input port');

    hError = comm.ErrorRate('ResetInputPort',true);

 
    ofdmInfo = info(hOFDMmod)
    
    numDC = ofdmInfo.DataInputSize(1)
   
    frameSize = [k*numDC 1];

   
    EbNoVec = (0:33)';
    snrVec = EbNoVec + 10*log10(k) + 10*log10(numDC/numSC);
   
    berVec2 = zeros(length(EbNoVec),3);
    errorStats = zeros(1,3);
    
    for m = 1:length(EbNoVec)
        snr = snrVec(m);

        while errorStats(2) <= maxBitErrors && errorStats(3) <= maxNumBits
            dataIn = randi([0,1],frameSize);              % Generate binary data
            qpskTx = step(hQPSKMod,dataIn);               % Apply QPSK modulation
            txSig = step(hOFDMmod,qpskTx);                % Apply OFDM modulation
            powerDB = 10*log10(var(txSig));               % Calculate Tx signal power
            noiseVar = 10.^(0.1*(powerDB-snr));           % Calculate the noise variance
            rxSig = step(hChan,txSig,noiseVar);           % Pass the signal through a noisy channel
            qpskRx = step(hOFDMdemod,rxSig);              % Apply OFDM demodulation
            dataOut = step(hQPSKDemod,qpskRx);            % Apply QPSK demodulation
            errorStats = step(hError,dataIn,dataOut,0);   % Collect error statistics
        end

        berVec2(m,:) = errorStats;                         % Save BER data
        errorStats = step(hError,dataIn,dataOut,1);       % Reset the error rate calculator
    end
   


    figure
    semilogy(EbNoVec,berVec1(:,1),'--*')
    hold on
    semilogy(EbNoVec,berVec5(:,1),'--*')
    hold on
    semilogy(EbNoVec,berVec4(:,1),'--*')
    hold on
    semilogy(EbNoVec,berVec(:,1),'--*')
    hold on
    semilogy(EbNoVec,berVec3(:,1),'--*')
    hold on
    semilogy(EbNoVec,berVec2(:,1),'--*')
    hold on
    




EsN0dB 	= [0:33]; % symbol to noise ratio
M=16; % 16QAM/64QAM and 256 QAM     
	k      = sqrt(1/((2/3)*(M-1))); 
	simSer1(1,:) = compute_symbol_error_rate(EsN0dB, M(1));
	
	semilogy(EsN0dB,simSer1(1,:),'r*');
M=64;
	k      = sqrt(1/((2/3)*(M-1))); 
	simSer2(2,:) = compute_symbol_error_rate(EsN0dB, M);
	
	hold on
	semilogy(EsN0dB,simSer2(2,:),'b*');
M=256;
	k      = sqrt(1/((2/3)*(M-1))); 
	simSer3(3,:) = compute_symbol_error_rate(EsN0dB, M);
	
	hold on
	semilogy(EsN0dB,simSer3(3,:),'g*');


legend('BPSK','QPSK','8-PSK','DBPSK','DQPSK','8-DPSK','16-QAM','64-QAM','256-QAM','Location','SouthWest')
    title('SNR vs BER for BPSK/QPSK/8-PSK/DBPSK/DQPSK/8-DPSK/16,64,256-QAM MIMO OFDM over AWGN')
    xlabel('Eb/No (dB)')
    ylabel('Bit Error Rate')
    grid on
    hold off
return ;

function [simSer] = compute_symbol_error_rate(EsN0dB, M);

nFFT = 64; % fft size
nDSC = 52; % number of data subcarriers
nConstperOFDMsym = 52; % number of bits per OFDM symbol (same as the number of subcarriers for BPSK)
nOFDMsym = 10^4; % number of ofdm symbols

k = sqrt(1/((2/3)*(M-1))); % normalizing factor
m = [1:sqrt(M)/2]; % alphabets
alphaMqam = [-(2*m-1) 2*m-1]; 

EsN0dB_eff = EsN0dB  + 10*log10(nDSC/nFFT) + 10*log10(64/80); % accounting for the used subcarriers and cyclic prefix

for ii = 1:length(EsN0dB)

   ipMod = randsrc(1,nConstperOFDMsym*nOFDMsym,alphaMqam) + j*randsrc(1,nConstperOFDMsym*nOFDMsym,alphaMqam);
   ipMod_norm = k*reshape(ipMod,nConstperOFDMsym,nOFDMsym).'; % grouping into multiple symbolsa

   xF = [zeros(nOFDMsym,6) ipMod_norm(:,[1:nConstperOFDMsym/2]) zeros(nOFDMsym,1) ipMod_norm(:,[nConstperOFDMsym/2+1:nConstperOFDMsym]) zeros(nOFDMsym,5)] ;
    
   xt = (nFFT/sqrt(nDSC))*ifft(fftshift(xF.')).';

   xt = [xt(:,[49:64]) xt];

   xt = reshape(xt.',1,nOFDMsym*80);

   nt = 1/sqrt(2)*[randn(1,nOFDMsym*80) + j*randn(1,nOFDMsym*80)];

   % Adding noise, the term sqrt(80/64) is to account for the wasted energy due to cyclic prefix
   yt = sqrt(80/64)*xt + 10^(-EsN0dB_eff(ii)/20)*nt;

   yt = reshape(yt.',80,nOFDMsym).'; % formatting the received vector into symbols
   yt = yt(:,[17:80]); % removing cyclic prefix

   yF = (sqrt(nDSC)/nFFT)*fftshift(fft(yt.')).'; 
   yMod = sqrt(64/80)*yF(:,[6+[1:nConstperOFDMsym/2] 7+[nConstperOFDMsym/2+1:nConstperOFDMsym] ]); 

   y_re = real(yMod)/k;
   y_im = imag(yMod)/k;

   ipHat_re = 2*floor(y_re/2)+1;
   ipHat_re(find(ipHat_re>max(alphaMqam))) = max(alphaMqam);
   ipHat_re(find(ipHat_re<min(alphaMqam))) = min(alphaMqam);

   ipHat_im = 2*floor(y_im/2)+1;
   ipHat_im(find(ipHat_im>max(alphaMqam))) = max(alphaMqam);
   ipHat_im(find(ipHat_im<min(alphaMqam))) = min(alphaMqam);
    
   ipHat = ipHat_re + j*ipHat_im; 

   % converting to vector 
   ipHat_v = reshape(ipHat.',nConstperOFDMsym*nOFDMsym,1).';

   % counting the errors
   nErr(ii) = size(find(ipMod - ipHat_v ),2);

end
simSer = nErr/(nOFDMsym*nConstperOFDMsym);

return;