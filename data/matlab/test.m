%% 参数设置
% 系统参数配置
snr = 15;                                                 % 信噪比
numSubc = 64;                                             % FFT 长度
numGuardBands = [6;6];                                    % 左右保护带
numPilot = 4;                                             % 每根天线的导频子载波
numTx = 2;                                                % 发射天线数量
numRx = 2;                                                % 接收天线数量
numSym = 14;                                              % 每帧 OFDM 符号数
numStream = 2;                                            % 数据流个数
cpLength = 16;                                            % 循环前缀长度

% 调制参数配置
M = 4;                                                    % QPSK 调制（M=4）
bitsPerSym = log2(M);                                     % 每符号比特数

% 信道编码参数
trellis = poly2trellis(7, [171 133]);                     % 卷积码结构
codeRate = 1/2;                                           % 编码速率

% 信道模型配置
sampleRate = 15.36e6;                                     % 采样率
pathDelays = [0 0.5e-6];                                  % 路径时延
averagePathGains = [0 -2];                                % 平均路径增益
maxDopplerShift = 300;                                    % 最大多普勒频移

% 信道估计配置
CEC.pilotAverage = 'UserDefined';
CEC.freqWindow = 3;
CEC.timeWindow = 3;
CEC.interpType = 'linear';
CEC.algorithm = 'ls';

% 导频子载波配置
pilotIndicesAnt1 = [7; 26; 40; 57]; % 天线 1 导频索引
pilotIndicesAnt2 = [8; 27; 41; 58]; % 天线 2 导频索引
pilotIndices = zeros(numPilot, numSym, numTx);
pilotIndices(:, :, 1) = repmat(pilotIndicesAnt1, 1, numSym);
pilotIndices(:, :, 2) = repmat(pilotIndicesAnt2, 1, numSym);

% 数据子载波配置
dataIndices = setdiff((numGuardBands(1)+1):(numSubc-numGuardBands(2)),unique(pilotIndices));
numDataSubc = length(dataIndices);
numFrameSymbols = numDataSubc * numSym * numTx;           % 总数据符号数

% OFDM解调器
ofdmDemod = comm.OFDMDemodulator('FFTLength', numSubc, ...
                                  'NumGuardBandCarriers', numGuardBands, ...
                                  'NumSymbols', numSym, ...
                                  'PilotOutputPort', true, ...
                                  'PilotCarrierIndices', pilotIndices, ...
                                  'CyclicPrefixLength', cpLength, ...
                                  'NumReceiveAntennas', numRx);
ofdmMod = comm.OFDMModulator(ofdmDemod);

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
    'PathGainsOutputPort', true);

%% 数据发送与接收
% 生成信息比特
numInfoBits = numFrameSymbols * bitsPerSym * codeRate;     % 计算信息比特数
infoBits = randi([0 1], numInfoBits, 1);                   % 生成随机信息比特

% 信道编码
encodedBits = convenc(infoBits, trellis);                  % 卷积编码

% 调制
txSymStream = pskmod(encodedBits, M, 'InputType', 'bit');  % QPSK调制

% 重塑数据符号维度
dataSignal = reshape(txSymStream, numDataSubc, numSym, numTx);

% 导频符号生成
pilotSignal = repmat(1+1i, numPilot, numSym, numTx);

% OFDM 调制
txSignal = ofdmMod(dataSignal, pilotSignal);

% 通过信道模型
[airSignal, pathGains] = mimoChannel(txSignal);

% 添加噪声
[rxSignal, noiseVar] = awgn(airSignal, snr, "measured");

% OFDM 解调 
[rxDataSignal, rxPilotSignal] = ofdmDemod(rxSignal);

%% 信道估计与均衡
% 完美信道估计
mimoChannelInfo = info(mimoChannel);
pathFilters = mimoChannelInfo.ChannelFilterCoefficients;
toffset = mimoChannelInfo.ChannelFilterDelay;
hPerfect = ofdmChannelResponse(pathGains, pathFilters, numSubc, cpLength, dataIndices, toffset);

% LS信道估计
hEst = channelEstimate(rxPilotSignal, pilotSignal, dataIndices, pilotIndices, CEC);

% MMSE 均衡
eqSignalMMSE = myMMSEequalize(hEst, rxDataSignal, noiseVar);
eqSignalMMSE = reshape(eqSignalMMSE, [], 1);

% 计算LLR并译码
llr = pskdemod(eqSignalMMSE, M, 'OutputType', 'approxllr', 'NoiseVariance', noiseVar);
decodedBits = vitdec(llr, trellis, 34, 'trunc', 'unquant'); % 维特比译码

%% 性能评估
errorRate = comm.ErrorRate;
BER = errorRate(infoBits, decodedBits(1:length(infoBits))); % 计算BER
fprintf('\nBER = %d from %d errors in %d bits\n', BER);


%% 自定义函数
function [H_est] = channelEstimate(rxPilotSignal,refPilotSignal, dataIndices, pilotIndices, CEC)
    % MIMO-OFDM 信道估计函数
    % 输入：
    %   rxPilotSignal: 接收导频信号(numPilotSubc x numSym x numTx x numRx)
    %   refPilotSignal: 参考导频信号 (numPilotSubc x numSym x numTx)
    %   dataIndices: 数据符号索引
    %   pilotIndices: 导频符号索引(numPilotSubc x numSym x numTx)
    %   CEC.algorithm: 估计算法 ('LS' 或 'MMSE')
    %   CEC.interpType: 插值类型 ('nearest', 'linear', 'cubic', 'spline')
    %   CEC.freqWindow: 频域平均窗口大小;
    %   CEC.timeWindow: 时域平均窗口大小;
    %   
    % 输出：
    %   H_est: 估计的信道响应矩阵 (numSubc x numSym x numTx x numRx)

    % 提取信号维度
    numDataSubc = length(dataIndices);
    [~, numSym, numTx, numRx] = size(rxPilotSignal);

    % 初始化估计的信道响应矩阵和噪声功率
    H_est = zeros(numDataSubc, numSym, numTx, numRx);

    for tx = 1:numTx
        for rx = 1:numRx
            % 获取当前发射 - 接收天线对的导频信号
            pilotRxSignal = rxPilotSignal(:,:,tx,rx);
            pilotRefSignal = refPilotSignal(:, :, tx);
            H_ls = pilotRxSignal ./ pilotRefSignal; 
            % 导频平均
            H_avg = pilotAveraging(H_ls, CEC.freqWindow, CEC.timeWindow);
            % 信道插值
            [X, Y] = meshgrid(1:numSym, dataIndices);
            [Xp, Yp] = meshgrid(1:numSym,squeeze(pilotIndices(:,1,tx)));
            H_est(:, :, tx, rx)  = griddata(Xp, Yp, H_avg, X, Y);
        end
    end
end

function H_avg = pilotAveraging(H_ls, freqWindow, timeWindow)
    [numPilotSubc, numSym] = size(H_ls);
    H_avg = zeros(size(H_ls));

    halfFreqWindow = floor(freqWindow / 2);
    halfTimeWindow = floor(timeWindow / 2);

    for sym = 1:numSym
        for subc = 1:numPilotSubc
            startSubc = max(1, subc - halfFreqWindow);
            endSubc = min(numPilotSubc, subc + halfFreqWindow);
            startSym = max(1, sym - halfTimeWindow);
            endSym = min(numSym, sym + halfTimeWindow);

            window = H_ls(startSubc:endSubc, startSym:endSym);
            H_avg(subc, sym) = mean(window(:));
        end
    end
end

function [out, csi] = myZFequalize(H, rxsignal)
% myZFequalize 自定义零迫（ZF）均衡函数
%
% 输入：
%   H: 信道矩阵，尺寸为 [nsc, nsym, ntx, nrx]
%      - nsc: 子载波数量
%      - nsym: OFDM 符号数量
%      - ntx: 发射天线数
%      - nrx: 接收天线数
%
%   rxsignal: 接收信号，尺寸为 [nsc, nsym, nrx]
%
% 输出：
%   out: 均衡后估计的发送符号，尺寸为 [nsc, nsym, ntx]
%   csi: 软信道状态信息，尺寸为 [nsc, nsym, ntx]
%
% 算法说明：
% 对于每个子载波和每个 OFDM 符号，
% 1. 从 H 中提取对应的信道矩阵，原始尺寸为 [ntx, nrx]，
%    转置后变为 [nrx, ntx]，符合 y = H * x 的模型。
% 2. 计算伪逆： x_hat = pinv(H_sub) * y
% 3. 同时计算 CSI，例如取 (H_sub^H * H_sub) 的对角线并取倒数，
%    作为各个发射分量的信道“质量”指标。

% 获取各维度大小
[nsc, nsym, ntx, nrx] = size(H);

% 初始化输出变量
out = zeros(nsc, nsym, ntx);
csi = zeros(nsc, nsym, ntx);

% 对每个子载波和每个 OFDM 符号进行循环
for i = 1:nsc
    for j = 1:nsym
        % 提取当前子载波和符号的信道矩阵
        % H(i,j,:,:) 的尺寸为 [ntx, nrx]
        % 为了使其符合 y = H_sub * x 模型，将其转置为 [nrx, ntx]
        H_sub = squeeze(H(i, j, :, :)).';  % 结果尺寸为 [nrx, ntx]
        
        % 提取当前子载波和符号的接收信号
        % rxsignal(i,j,:) 的尺寸为 [nrx, 1]
        y = squeeze(rxsignal(i, j, :));
        
        % 计算零迫均衡
        % 若 H_sub 为满列秩，则 pinv(H_sub) = inv(H_sub' * H_sub) * H_sub'
        x_hat = pinv(H_sub) * y;
        out(i, j, :) = x_hat;
        
        % 计算 CSI 信息：
        % 这里简单计算 H_sub^H * H_sub，然后取其对角线（每个发射分量的能量）
        % 并取倒数，作为信道质量指标。注意防止除零错误。
        Hhh = H_sub' * H_sub;    % 尺寸为 [ntx, ntx]
        diag_val = diag(Hhh);     % 取对角线，得到 [ntx, 1] 向量
        % 防止除数为 0
        diag_val(diag_val < eps) = eps;
        csi(i, j, :) = 1 ./ diag_val;
    end
end

end

function [out, csi] = myMMSEequalize(H, rxsignal, noiseVar)
% myMMSEequalize 自定义 MMSE 均衡器
%
% 输入：
%   H: 信道矩阵，尺寸为 [nsc, nsym, ntx, nrx]
%      - nsc: 子载波数量
%      - nsym: OFDM 符号数量
%      - ntx: 发射天线数
%      - nrx: 接收天线数
%
%   rxsignal: 接收信号，尺寸为 [nsc, nsym, nrx]
%
%   noiseVar: 噪声方差，可以为标量或行向量（长度为 nsym）
%
% 输出：
%   out: 均衡后估计的发送符号，尺寸为 [nsc, nsym, ntx]
%   csi: 软信道状态信息，尺寸为 [nsc, nsym, ntx]
%
% 算法说明：
% 对于每个子载波 i 和每个 OFDM 符号 j：
% 1. 提取信道矩阵 H(i,j,:,:)（尺寸 [ntx, nrx]），转置后变为 H_sub (nrx×ntx)
%    对于 y = H_sub * x 模型，y 为 rxsignal(i,j,:)（nrx×1），x 为发射信号 (ntx×1)
% 2. 计算 MMSE 均衡器：
%       x_hat = inv( H_sub^H * H_sub + noiseVar * I ) * H_sub^H * y
% 3. 同时可以计算 CSI，比如取对角线的倒数：
%       csi(i,j,:) = 1./diag(H_sub^H * H_sub + noiseVar * I)
%
% 注意：为了防止除零，需要对非常小的对角线值加以限制。

% 获取尺寸
[nsc, nsym, ntx, nrx] = size(H);

% 初始化输出
out = zeros(nsc, nsym, ntx);
csi = zeros(nsc, nsym, ntx);

% 如果 noiseVar 为标量，则扩展为 nsym 的行向量
if isscalar(noiseVar)
    noiseVarVec = repmat(noiseVar, 1, nsym);
else
    noiseVarVec = noiseVar;
end

% 对每个子载波和每个 OFDM 符号循环
for i = 1:nsc
    for j = 1:nsym
        % 提取当前子载波和符号的信道矩阵，尺寸为 [ntx, nrx]
        H_ij = squeeze(H(i, j, :, :)); % [ntx, nrx]
        % 转置 H，使其符合 y = H_sub * x 模型，得到 H_sub 尺寸 [nrx, ntx]
        H_sub = H_ij.'; 
        
        % 提取当前子载波和符号的接收信号 y，尺寸为 [nrx, 1]
        y = squeeze(rxsignal(i, j, :));
        
        % 构造单位矩阵，尺寸为 [ntx, ntx]
        I = eye(ntx);
        
        % 计算 MMSE 均衡器矩阵：
        %   W = inv( H_sub^H * H_sub + noiseVar * I ) * H_sub^H
        % 注意：noiseVar 对应当前符号 j
        W = pinv(H_sub' * H_sub + noiseVarVec(j) * I) * H_sub';
        
        % 估计发送信号
        x_hat = W * y;
        out(i, j, :) = x_hat;
        
        % 计算 CSI：这里取 H_sub^H * H_sub + noiseVar * I 的对角线元素，并取倒数
        % 对角线元素代表各个发射天线分量的“有效”增益，噪声项可视为权重修正
        Hhh = H_sub' * H_sub + noiseVarVec(j) * I;
        diagVal = diag(Hhh);
        % 防止除零
        diagVal(diagVal < eps) = eps;
        csi(i, j, :) = 1 ./ diagVal;
    end
end

end
