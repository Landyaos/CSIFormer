%% 系统配置 (2x2 MIMO)
carrier = nrCarrierConfig('NSizeGrid',52,'SubcarrierSpacing',15,'CyclicPrefix','Normal');
carrier.NTxAnts = 2;
carrier.NRxAnts = 2;

pdsch = nrPDSCHConfig;
pdsch.NumLayers = 2;
pdsch.DMRS.DMRSPortSet = [0 1];
pdsch.DMRS.DMRSConfigurationType = 2;

%% 信道模型配置
channel = nrCDLChannel('DelayProfile','CDL-A','MaximumDopplerShift',100);
channel.NumTransmitAntennas = carrier.NTxAnts;
channel.NumReceiveAntennas = carrier.NRxAnts;

%% 仿真参数
snrValues = 0:5:30;
numFrames = 20;
berResults = zeros(length(snrValues), 4); % 存储4种算法的BER

%% 主仿真循环
for snrIdx = 1:length(snrValues)
    SNR = snrValues(snrIdx);
    fprintf('Processing SNR=%ddB...\n', SNR);
    
    totalErrors = zeros(4,1);
    totalBits = 0;
    
    for frameIdx = 1:numFrames
        %% 发射端处理
        % 生成发送数据 (标记点1)
        txBits = randi([0 1], pdsch.TransportBlockSize, 1);
        
        % PDSCH处理
        [txSymbols, pdschIndices] = nrPDSCH(carrier, pdsch, txBits);
        
        % 生成导频 (标记点2)
        dmrsSymbols = nrPDSCHDMRS(carrier, pdsch);
        dmrsIndices = nrPDSCHDMRSIndices(carrier, pdsch);
        
        % 资源映射 (标记点3)
        txGrid = nrResourceGrid(carrier);
        txGrid(pdschIndices) = txSymbols;
        txGrid(dmrsIndices) = dmrsSymbols;
        
        % OFDM调制
        txWaveform = nrOFDMModulate(carrier, txGrid);
        
        %% 信道传输
        reset(channel);
        [rxWaveform, pathGains, pathDelays] = channel(txWaveform);
        rxWaveform = awgn(rxWaveform, SNR, 'measured');
        
        %% 接收端处理
        rxGrid = nrOFDMDemodulate(carrier, rxWaveform);
        
        % 理想CSI获取 (标记点4)
        [~, pathFilters] = getPathFilters(channel);
        idealH = nrPerfectChannelEstimate(carrier, pathGains, pathDelays, pathFilters);
        
        %% 信道估计方法比较
        % 方法1: 完美CSI + MMSE均衡
        eqGrid1 = nrEqualizeMMSE(rxGrid, idealH, 10^(-SNR/10));
        [estBits1,~] = nrPDSCHDecode(carrier, pdsch, eqGrid1(pdschIndices));
        
        % 方法2: LS估计 + ZF均衡
        estH2 = nrChannelEstimate(rxGrid, dmrsIndices, dmrsSymbols);
        eqGrid2 = rxGrid ./ estH2; % ZF均衡
        [estBits2,~] = nrPDSCHDecode(carrier, pdsch, eqGrid2(pdschIndices));
        
        % 方法3: MMSE估计 + MMSE均衡
        noiseVar = 10^(-SNR/10);
        estH3 = nrChannelEstimate(rxGrid, dmrsIndices, dmrsSymbols, 'NoiseVariance',noiseVar);
        eqGrid3 = nrEqualizeMMSE(rxGrid, estH3, noiseVar);
        [estBits3,~] = nrPDSCHDecode(carrier, pdsch, eqGrid3(pdschIndices));
        
        % 方法4: LS估计 + MMSE均衡
        eqGrid4 = nrEqualizeMMSE(rxGrid, estH2, noiseVar);
        [estBits4,~] = nrPDSCHDecode(carrier, pdsch, eqGrid4(pdschIndices));
        
        %% BER计算
        totalErrors(1) = totalErrors(1) + sum(estBits1 ~= txBits);
        totalErrors(2) = totalErrors(2) + sum(estBits2 ~= txBits);
        totalErrors(3) = totalErrors(3) + sum(estBits3 ~= txBits);
        totalErrors(4) = totalErrors(4) + sum(estBits4 ~= txBits);
        totalBits = totalBits + length(txBits);
    end
    
    berResults(snrIdx,:) = totalErrors / totalBits;
end

%% BER可视化
figure;
semilogy(snrValues, berResults(:,1), 'ro-', 'LineWidth',2); hold on;
semilogy(snrValues, berResults(:,2), 'bs--', 'LineWidth',1.5);
semilogy(snrValues, berResults(:,3), 'g^-.', 'LineWidth',1.5);
semilogy(snrValues, berResults(:,4), 'md:', 'LineWidth',1.5);
xlabel('SNR (dB)'); ylabel('BER');
title('不同信道估计与均衡算法性能比较');
legend('完美CSI+MMSE','LS+ZF','MMSE+MMSE','LS+MMSE');
grid on;