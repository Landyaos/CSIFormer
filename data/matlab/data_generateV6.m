clear;
clc;

%% 参数设置
% 系统参数配置
numSubc = 256;                                            % FFT 长度
numGuardBands = [16;15];                                  % 左右保护带
numPilot = (numSubc-sum(numGuardBands)-1)/4;              % 每根天线的导频子载波
numTx = 2;                                                % 发射天线数量
numRx = 2;                                                % 接收天线数量
numSym = 14;                                              % 每帧 OFDM 符号数
numStream = 2;                                            % 数据流个数
cpLength = numSubc/4;                                            % 循环前缀长度

% 调制参数配置
M = 4;                                                    % QPSK 调制（M=4）

% 信道模型配置
sampleRate = 15.36e6;                                        % 采样率
pathDelays = [0, 30, 70, 90, 110, 190, 410] * 1e-9;       % 路径时延
averagePathGains = [0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8];                             % 平均路径增益
maxDopplerShift = 5.5;                                    % 最大多普勒频移

% 信号导频分布配置
validSubcIndices = setdiff((numGuardBands(1)+1):(numSubc-numGuardBands(2)), numSubc/2+1);
numValidSubc = length(validSubcIndices);

% 导频子载波配置
pilotIndicesAnt1 = (numGuardBands(1)+1:4:numSubc-numGuardBands(2))';                        % 发射天线1的导频子载波位置
pilotIndicesAnt2 = (numGuardBands(1)+2:4:numSubc-numGuardBands(2))';                        % 发射天线2的导频子载波位置
pilotIndicesAnt2(end) = pilotIndicesAnt1(end)-1;
pilotIndicesAnt1 = pilotIndicesAnt1(pilotIndicesAnt1~=numSubc/2+1);                         % DC子载波不允许设为导频，因此去除
pilotIndicesAnt2 = pilotIndicesAnt2(pilotIndicesAnt1~=numSubc/2+1);                         % DC子载波不允许设为导频，因此去除

% 构造 PilotCarrierIndices (3D 矩阵, NPilot-by-NSym-by-NT)
pilotIndices = zeros(numPilot, numSym, numTx);
pilotIndices(:, :, 1) = repmat(pilotIndicesAnt1, 1, numSym); % 天线 1
pilotIndices(:, :, 2) = repmat(pilotIndicesAnt2, 1, numSym); % 天线 2

% 数据子载波配置
dataIndices = setdiff((numGuardBands(1)+1):(numSubc-numGuardBands(2)),[unique(pilotIndices); numSubc/2+1]);
numDataSubc = length(dataIndices);
numFrameSymbols = numDataSubc * numSym * numTx;

% OFDM解调器
ofdmDemod = comm.OFDMDemodulator('FFTLength', numSubc, ...
                                  'NumGuardBandCarriers', numGuardBands, ...
                                  'RemoveDCCarrier', true,...
                                  'NumSymbols', numSym, ...
                                  'PilotOutputPort', true, ...
                                  'PilotCarrierIndices', pilotIndices, ...
                                  'CyclicPrefixLength', cpLength, ...
                                  'NumReceiveAntennas', numRx);
% OFDM调制器 
ofdmMod = comm.OFDMModulator('FFTLength', numSubc, ...
                             'NumGuardBandCarriers', numGuardBands, ...
                             'InsertDCNull', true,...
                             'NumSymbols', numSym, ...
                             'PilotInputPort', true, ...
                             'PilotCarrierIndices', pilotIndices, ...
                             'CyclicPrefixLength', cpLength, ...
                             'NumTransmitAntennas', numTx);


%% 数据集采集
numFrame = 2;
snrValues = 20:5:30;
datasetPath = {'../raw/trainDataV4.mat','../raw/valDataV4.mat'};
datasetConfig = [12000,2000];


error = zeros(length(snrValues),3);
for datasetIdx = 1:length(datasetPath)
    
    snrDatasetSize = datasetConfig(datasetIdx);
    datasetCapacity = snrDatasetSize * length(snrValues);
    
    txSignalData = zeros(datasetCapacity, numValidSubc, numSym, numTx, 2);
    rxSignalData = zeros(datasetCapacity, numValidSubc, numSym, numRx, 2);
    csiLSData = zeros(datasetCapacity, numValidSubc, numSym, numTx, numRx, 2);
    csiLabelData = zeros(datasetCapacity, numValidSubc, numSym, numTx, numRx, 2);
    csiPreData = zeros(datasetCapacity, numFrame, numValidSubc, numSym, numTx, numRx, 2);
    
    csiPreTemp = zeros(numFrame+1, numValidSubc, numSym, numTx, numRx, 2);
    for snrIdx = 1:length(snrValues)
        snrIdx
        snr = snrValues(snrIdx);
        minSeed = 0;
        maxSeed = 2^32 - 1;   % 3.4625e+09 2.4608e+09 4.0359e+09 2.7777e+09
        
        seed = randi([minSeed, maxSeed]);
        seed
        % 信道模型
        mimoChannel = comm.MIMOChannel(...
            'SampleRate', sampleRate, ...
            'SpatialCorrelationSpecification', 'None',...
            'PathDelays', pathDelays, ...
            'AveragePathGains', averagePathGains, ...
            'MaximumDopplerShift', maxDopplerShift, ...
            'NumTransmitAntennas', numTx, ...
            'NumReceiveAntennas', numRx, ...
            'FadingDistribution', 'Rayleigh', ...
            'PathGainsOutputPort', true,...
            'RandomStream', 'mt19937ar with seed', ...  % 使用固定种子的随机数流
            'Seed', seed);   % 开启路径增益输出

        mimoChannelInfo = info(mimoChannel);
        pathFilters = mimoChannelInfo.ChannelFilterCoefficients;
        toffset = mimoChannelInfo.ChannelFilterDelay;        
        for frame = -1:snrDatasetSize
            % 数据符号生成
            txSymStream = randi([0 M-1], numFrameSymbols, 1); 
            dataSignal = pskmod(txSymStream, M);  % 调制后的符号为复数形式
            dataSignal = reshape(dataSignal, numDataSubc, numSym, numTx);
            pilotSignal = repmat(1+1i, numPilot, numSym, numTx);
            
            originSignal = zeros(numSubc, numSym, numTx);
            originSignal(dataIndices,:,:) = dataSignal;
            for tx = 1:numTx
                originSignal(pilotIndices(:,1,tx),:,tx) = pilotSignal(:, :, tx);
            end    

            % OFDM 调制
            txSignal = ofdmMod(dataSignal, pilotSignal);
            % 通过信道模型获取接收信号和路径增益[总样本数, N_path, numTransmitAntennas, numReceiveAntennas]
            [airSignal, pathGains] = mimoChannel(txSignal);
            % 去滤波器时延
            airSignal = [airSignal((toffset+1):end,:); zeros(toffset,2)];
            % 噪声
            [airSignal, noiseVar] = awgn(airSignal, snr, "measured");
            % OFDM 解调
            [rxDataSignal, rxPilotSignal] = ofdmDemod(airSignal);

            % 完美CSI矩阵Nsc x Nsym x Nt x Nr
            h = ofdmChannelResponse(pathGains, pathFilters, numSubc, cpLength, validSubcIndices, toffset);
            % 计算导频处信道估计
            csiLS = zeros(numSubc, numSym, numTx, numRx);
            for tx = 1:numTx
                for rx = 1:numRx
                    csiLS(pilotIndices(:,1,tx),:,tx,rx) = rxPilotSignal(:,:,tx,rx) ./ pilotSignal(:, :, tx);
                end
            end

            csiPreTemp(1,:,:,:,:,:) = csiPreTemp(2,:,:,:,:,:);
            csiPreTemp(2,:,:,:,:,:) = csiPreTemp(3,:,:,:,:,:);
            csiPreTemp(3,:,:,:,:,1) = real(csiLS(validSubcIndices,:,:,:));
            csiPreTemp(3,:,:,:,:,2) = imag(csiLS(validSubcIndices,:,:,:));

            if frame < 1
                continue;
            end
            % 接收信号
            finalSignal = zeros(numSubc, numSym, numRx);
            finalSignal(dataIndices,:,:) = rxDataSignal;
            for rx = 1:numRx
                for tx = 1:numTx
                    finalSignal(pilotIndices(:,1,tx),:,rx) = rxPilotSignal(:,:,tx,rx);
                end
            end

            
            %% 数据保存
            dataIdx = snrDatasetSize * (snrIdx-1) + frame;

            % csi_pre
            csiPreData(dataIdx,:,:,:,:,:,:) = csiPreTemp(1:2,:,:,:,:,:);
            % csi_label
            csiLabelData(dataIdx,:,:,:,:,1) = real(h);
            csiLabelData(dataIdx,:,:,:,:,2) = imag(h);
            % csi_ls
            csiLSData(dataIdx,:,:,:,:,1) = real(csiLS(validSubcIndices,:,:,:));
            csiLSData(dataIdx,:,:,:,:,2) = imag(csiLS(validSubcIndices,:,:,:));
            % 发送信号（实部和虚部分离）
            txSignalData(dataIdx,:,:,:,1) = real(originSignal(validSubcIndices, :, :));
            txSignalData(dataIdx,:,:,:,2) = imag(originSignal(validSubcIndices, :, :));
            % 接收信号（实部和虚部分离）
            rxSignalData(dataIdx,:,:,:,1) = real(finalSignal(validSubcIndices, :, :));
            rxSignalData(dataIdx,:,:,:,2) = imag(finalSignal(validSubcIndices, :, :));
            
        end

    end  
    disp('save data ...')
    % 保存批量数据到文件
    save(datasetPath{datasetIdx}, ...
        'csiLSData',...
        'csiPreData',...
        'csiLabelData', ...
        'txSignalData',...
        'rxSignalData',...
        '-v7.3');
end


