generateAndSaveDataset(10000, 'mimo_ofdm_dataset1.mat');
function generateAndSaveDataset(numSamples, savePath)
    % 参数设置
    numSubcarriers = 64;                                     
    numGuardBands = [6; 6];                                    
    numPilots = 4;                                            
    numTransmitAntennas = 2;                                  
    numReceiveAntennas = 2;                                  
    numSymbols = 14;                                          
    cpLength = 16;                                            
    modulationOrder = 4;                                      
    bitsPerSymbol = log2(modulationOrder);
    snr = 10;
    sampleRate = 15.36e6;                                     
    pathDelays = [0 0.5e-6 1.2e-6];                           
    averagePathGains = [0 -2 -5];                             
    maxDopplerShift = 50;                                     
    
    % 初始化存储变量
    inputs = {};
    labels = {};

    for i = 1:numSamples
        % 生成随机比特流和数据符号
        validSubcarrierRange = (numGuardBands(1)+1):(numSubcarriers-numGuardBands(2));
        pilotIndicesAntenna1 = [12; 26; 40; 54];
        pilotIndicesAntenna2 = [13; 27; 41; 55];
        pilotIndices = zeros(numPilots, numSymbols, numTransmitAntennas);
        pilotIndices(:, :, 1) = repmat(pilotIndicesAntenna1, 1, numSymbols); 
        pilotIndices(:, :, 2) = repmat(pilotIndicesAntenna2, 1, numSymbols); 
        pilotQPSKSymbols = [1+1i, 1+1i, 1+1i, 1+1i];
        pilotSymbols = pilotQPSKSymbols(randi(length(pilotQPSKSymbols), numPilots, numSymbols, numTransmitAntennas));
        
        numDataSubcarriers = numSubcarriers - sum(numGuardBands) - (numPilots * numTransmitAntennas);
        numBits = numDataSubcarriers * numSymbols * bitsPerSymbol;
        originalBits = randi([0 1], numBits, 1);
        dataSymbols = pskmod(originalBits, modulationOrder, pi/4);
        dataSymbols = reshape(dataSymbols, numDataSubcarriers, numSymbols, numTransmitAntennas);
        
        % OFDM 调制
        ofdmMod = comm.OFDMModulator('FFTLength', numSubcarriers, ...
                                     'NumGuardBandCarriers', numGuardBands, ...
                                     'NumSymbols', numSymbols, ...
                                     'PilotInputPort', true, ...
                                     'PilotCarrierIndices', pilotIndices, ...
                                     'CyclicPrefixLength', cpLength, ...
                                     'NumTransmitAntennas', numTransmitAntennas);
        txSignal = ofdmMod(dataSymbols, pilotSymbols);

        % MIMO 信道模型
        mimoChannel = comm.MIMOChannel('SampleRate', sampleRate, ...
                                       'SpatialCorrelationSpecification', 'None',...
                                       'NumTransmitAntennas', numTransmitAntennas, ...
                                       'NumReceiveAntennas', numReceiveAntennas, ...
                                       'FadingDistribution', 'Rayleigh', ...
                                       'PathGainsOutputPort', true);
        [rxSignal, pathGains] = mimoChannel(txSignal);

        mimoChannelInfo = info(mimoChannel);
        pathFilters = mimoChannelInfo.ChannelFilterCoefficients;
        toffset = mimoChannelInfo.ChannelFilterDelay;
        h = ofdmChannelResponse(pathGains, pathFilters, numSubcarriers, cpLength, validSubcarrierRange, toffset);
        hReshaped = reshape(h, [], numTransmitAntennas, numReceiveAntennas);

        % 添加噪声
        rxSignal = awgn(rxSignal, snr, "measured");

        % OFDM 解调
        ofdmDemod = comm.OFDMDemodulator('FFTLength', numSubcarriers, ...
                                         'NumGuardBandCarriers', numGuardBands, ...
                                         'NumSymbols', numSymbols, ...
                                         'PilotOutputPort', true, ...
                                         'PilotCarrierIndices', pilotIndices, ...
                                         'CyclicPrefixLength', cpLength, ...
                                         'NumReceiveAntennas', numReceiveAntennas);
        [rxDataSymbols, rxPilotSymbols] = ofdmDemod(rxSignal);

        % 存储样本
        inputs{i} = {dataSymbols, pilotSymbols, pilotIndices, rxDataSymbols, rxPilotSymbols};
        labels{i} = hReshaped;
    end

    % 保存数据集
    save(savePath, 'inputs', 'labels','-v7.3');
    fprintf('Dataset saved to %s\n', savePath);
end
