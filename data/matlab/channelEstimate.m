% 测试参数设置
numSubc = 64; % 子载波数量
numSym = 16;  % OFDM符号数量
numTx = 2;    % 发射天线数量
numRx = 2;    % 接收天线数量

pilotIndicesAnt1 = 4:8:64;
pilotIndicesAnt2 = 8:8:64;

% 生成参考导频信号
refPilotSignal = ones(8, numSym, numTx); % 假设1/4的子载波为导频

% 生成数据和导频索引

pilotIndices = zeros(8, numSym, numTx);
pilotSignal = zeros(8, numSym, numTx);
for sym = 1:

disp(size(pilotIndices))
pilotCount = 1;
dataCount = 1;
for sym = 1:numSym
    for subc = 1:numSubc
        if mod(subc, 4) == 0
            pilotIndices(pilotCount, sym, :) = subc;
            pilotCount = pilotCount + 1;
        else
            dataIndices(dataCount, sym, :) = subc;
            dataCount = dataCount + 1;
        end
    end
end

disp(size(pilotIndices))

% 生成信道响应（模拟真实信道）
H_true = zeros(numSubc, numSym, numTx, numRx);
for tx = 1:numTx
    for rx = 1:numRx
        H_true(:, :, tx, rx) = randn(numSubc, numSym) + 1i * randn(numSubc, numSym);
    end
end

% 生成发送信号
txSignal = zeros(numSubc, numSym, numTx);
for tx = 1:numTx
    txSignal(:, :, tx) = randn(numSubc, numSym) + 1i * randn(numSubc, numSym);
end

% 通过信道传输并添加噪声
rxSignal = zeros(numSubc, numSym, numRx);
noisePower = 0.01;
for rx = 1:numRx
    for tx = 1:numTx
        rxSignal(:, :, rx) = rxSignal(:, :, rx) + H_true(:, :, tx, rx).* txSignal(:, :, tx);
    end
    rxSignal(:, :, rx) = rxSignal(:, :, rx) + sqrt(noisePower/2) * (randn(numSubc, numSym) + 1i * randn(numSubc, numSym));
end

% 配置信道估计参数
CEC.algorithm = 'LS'; % 选择LS算法
CEC.interp_type = 'linear'; % 线性插值
CEC.freqWindow = 3; % 频域平均窗口大小
CEC.timeWindow = 3; % 时域平均窗口大小

% 进行信道估计
[H_est, noiseVar] = channelEstimate1(rxSignal, refPilotSignal, dataIndices, pilotIndices, CEC);

% 计算均方误差（MSE）评估估计性能
mse = 0;
for tx = 1:numTx
    for rx = 1:numRx
        mse = mse + mean(mean(abs(H_true(:, :, tx, rx) - H_est(:, :, tx, rx)).^2));
    end
end
mse = mse / (numTx * numRx);
fprintf('使用LS算法的信道估计均方误差: %f\n', mse);

% 切换到MMSE算法
CEC.algorithm = 'MMSE';
[H_est_mmse, noiseVar_mmse] = channelEstimate(rxSignal, refPilotSignal, dataIndices, pilotIndices, CEC);

% 计算MMSE算法的均方误差
mse_mmse = 0;
for tx = 1:numTx
    for rx = 1:numRx
        mse_mmse = mse_mmse + mean(mean(abs(H_true(:, :, tx, rx) - H_est_mmse(:, :, tx, rx)).^2));
    end
end
mse_mmse = mse_mmse / (numTx * numRx);
fprintf('使用MMSE算法的信道估计均方误差: %f\n', mse_mmse);

function [H_est, noiseVar] = channelEstimate1(rxSignal, refPilotSignal, dataIndices, pilotIndices, CEC)
    % MIMO-OFDM 信道估计函数
    % 输入：
    %   rxSignal: 接收信号矩阵 (numSubc x numSym x numRx)
    %   refPilotSignal: 参考导频信号 (numPilotSubc x numSym x numTx)
    %   dataIndices: 数据符号索引 (numDataSubc x numSym x numTx)
    %   pilotIndices: 导频符号索引(numPilotSubc x numSym x numTx)
    %   CEC.algorithm: 估计算法 ('LS' 或 'MMSE')
    %   CEC.interpType: 插值类型 ('nearest', 'linear', 'cubic', 'spline')
    %   CEC.freqWindow: 频域平均窗口大小;
    %   CEC.timeWindow: 时域平均窗口大小;T
    %   
    % 输出：
    %   H_est: 估计的信道响应矩阵 (numSubc x numSym x numTx x numRx)
    %   noiseVar: 噪声平均功率

    % 提取信号维度
    [numSubc, numSym, numRx] = size(rxSignal);
    [~, ~, numTx] = size(refPilotSignal);

    % 初始化估计的信道响应矩阵和噪声功率
    H_est = zeros(numSubc, numSym, numTx, numRx);
    noiseVar = zeros(numTx, numRx);

    % 验证算法参数
    validAlgorithms = {'LS', 'MMSE'};
    if ~any(strcmpi(CEC.algorithm, validAlgorithms))
        error('Invalid algorithm. Must be one of: LS, MMSE');
    end

    % 验证插值类型
    validInterpTypes = {'nearest', 'linear', 'cubic','spline'};
    if ~any(strcmpi(CEC.interp_type, validInterpTypes))
        error('Invalid interpolation type. Must be one of: nearest, linear, cubic, spline');
    end

    for tx = 1:numTx
        for rx = 1:numRx
            % 获取当前发射 - 接收天线对的导频信号
            pilotRxSignal = rxSignal(pilotIndices(:, :, tx), :, rx);
            pilotRefSignal = refPilotSignal(:, :, tx);

            % 根据算法进行信道估计
            if strcmpi(CEC.algorithm, 'LS')
                % 最小二乘估计
                H_ls = pilotRxSignal./ pilotRefSignal;
            elseif strcmpi(CEC.algorithm, 'MMSE')
                % 简单假设信道自相关矩阵为单位矩阵
                R_hh = eye(size(pilotRefSignal, 1) * size(pilotRefSignal, 2));
                % 噪声功率谱密度暂时假设为1，实际需要估计
                noisePower = 1;
                R_nn = noisePower * eye(size(pilotRefSignal, 1) * size(pilotRefSignal, 2));
                R_yh = R_hh;
                mmseDenominator = R_yh' * inv(R_yh * R_yh' + R_nn) * R_yh;
                H_ls = mmseDenominator \ (pilotRxSignal(:));
                H_ls = reshape(H_ls, size(pilotRxSignal));
            end

            % 导频平均
            H_avg = pilotAveraging(H_ls, CEC.freqWindow, CEC.timeWindow);

            % 信道插值
            H_est(:, :, tx, rx) = channelInterpolation(H_avg, pilotIndices(:, :, tx), numSubc, numSym, CEC.interp_type);

            % 噪声估计
            dataRxSignal = rxSignal(dataIndices(:, :, tx), :, rx);
            dataRefSignal = refPilotSignal(:, :, tx);
            estimatedSignal = dataRefSignal.* H_est(dataIndices(:, :, tx), tx, rx);
            noise = dataRxSignal - estimatedSignal;
            noiseVar(tx, rx) = mean(abs(noise(:)).^2);
        end
    end

    % 计算平均噪声功率
    noiseVar = mean(noiseVar(:));
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

function H_interp = channelInterpolation(H_avg, pilotIndices, numSubc, numSym, interp_type)
    [X, Y] = meshgrid(1:numSubc, 1:numSym);
    Xi = pilotIndices(:, 1);
    Yi = pilotIndices(1, :);
    H_interp = griddata(Xi, Yi, H_avg, X, Y, interp_type);
end