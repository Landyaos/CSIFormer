% 示例参数
M = 64; % 子载波数量
N = 16; % OFDM 符号数量
NRxAnts = 2; % 接收天线数量
NTxAnts = 2; % 发射天线数量
RXGRID = randn(M, N, NRxAnts) + 1i * randn(M, N, NRxAnts); % 模拟接收信号

% 信道估计配置
CEC.PilotAverage = 'UserDefined';
CEC.FreqWindow = 3;
CEC.TimeWindow = 3;
CEC.InterpType = 'linear';

% 使用LS算法进行信道估计
[H_EST_ls, NoisePowerEst_ls] = mimoOFDMChannelEstimate(M, N, NRxAnts, NTxAnts, RXGRID, CEC, 'ls');

% 使用MMSE算法进行信道估计
[H_EST_mmse, NoisePowerEst_mmse] = mimoOFDMChannelEstimate(M, N, NRxAnts, NTxAnts, RXGRID, CEC,'mmse');
function [H_EST, NoisePowerEst] = mimoOFDMChannelEstimate(M, N, NRxAnts, NTxAnts, RXGRID, CEC, algorithm)
    % M: 子载波数量
    % N: OFDM 符号数量
    % NRxAnts: 接收天线数量
    % NTxAnts: 发射天线数量
    % RXGRID: 接收信号的时频网格
    % CEC: 信道估计配置参数
    % algorithm: 信道估计算法，'ls' 或'mmse'

    % 初始化估计信道和噪声功率
    H_EST = zeros(M, N, NRxAnts, NTxAnts);
    noiseVec = zeros(NRxAnts, NTxAnts);

    % 验证插值类型
    interptypes = {'nearest','linear','natural','cubic','v4','none'};
    if ~any(strcmpi(CEC.InterpType, interptypes))
        error('Invalid InterpType. Must be one of: nearest, linear, natural, cubic, v4, none');
    end

    % 验证算法参数
    validAlgorithms = {'ls','mmse'};
    if ~any(strcmpi(algorithm, validAlgorithms))
        error('Invalid algorithm. Must be one of: ls, mmse');
    end

    % 假设导频均匀分布在时频网格上
    pilotIndices = generatePilotIndices(M, N);

    for rxANT = 1:NRxAnts
        for txANT = 1:NTxAnts
            % 提取当前接收 - 发射天线对的导频信号并进行最小二乘估计
            [ls_estimates] = getLeastSquaresEstimates(RXGRID(:, :, rxANT), pilotIndices);

            % 根据算法进行信道估计
            if strcmpi(algorithm, 'ls')
                est = ls_estimates(3, :);
            elseif strcmpi(algorithm,'mmse')
                % 简单假设信道自相关矩阵为单位矩阵
                R_hh = eye(length(ls_estimates(3, :)));
                % 噪声功率谱密度近似为之前计算的噪声功率
                noisePower = mean(noiseVec(rxANT, txANT));
                R_nn = noisePower * eye(length(ls_estimates(3, :)));
                R_yh = R_hh;
                mmseDenominator = R_yh' * inv(R_yh * R_yh' + R_nn) * R_yh;
                est = mmseDenominator \ (ls_estimates(3, :));
            end

            % 将估计值放入P_EST
            P_EST = [ls_estimates(1:2, :); est];

            % 导频平均
            [P_EST, ScalingVec] = PilotAverage(CEC, H_EST(:, :, rxANT, txANT), P_EST);

            % 信道估计和插值
            if strcmpi(CEC.InterpType, 'none')
                H_EST(:, :, rxANT, txANT) = insertPilotEstimates(H_EST(:, :, rxANT, txANT), P_EST, pilotIndices);
            else
                H_EST(:, :, rxANT, txANT) = interpolateChannel(CEC, H_EST(:, :, rxANT, txANT), P_EST, pilotIndices);
            end

            % 计算噪声
            noise = ls_estimates(3, :) - P_EST(3, :);
            if strcmpi(CEC.PilotAverage, 'UserDefined')
                noise = sqrt(ScalingVec./(ScalingVec + 1)).*noise;
            end
            noiseVec(rxANT, txANT) = mean(noise.*conj(noise));
        end
    end

    % 计算整体噪声功率估计
    NoisePowerEst = mean(noiseVec, 'all', 'omitnan');
end

function [ls_estimates] = getLeastSquaresEstimates(RXGRID, pilotIndices)
    % 假设导频值为1（简化处理）
    refsym = ones(size(pilotIndices, 2), 1);
    rxsym = RXGRID(sub2ind(size(RXGRID), pilotIndices(1, :), pilotIndices(2, :)));
    p_est = rxsym./refsym;
    ls_estimates = [pilotIndices; p_est(:).'];
end

function [P_EST, ScalingVec] = PilotAverage(CEC, H_EST, P_EST)
    % 假设均匀的矩形平均窗口
    if strcmpi(CEC.PilotAverage, 'UserDefined')
        kernel = ones(CEC.FreqWindow, CEC.TimeWindow);
        reGrid = zeros(size(H_EST));
        reGrid(sub2ind(size(reGrid), P_EST(1, :), P_EST(2, :))) = P_EST(3, :);
        reGrid = conv2(reGrid, kernel,'same');
        tempGrid = zeros(size(reGrid));
        tempGrid(sub2ind(size(tempGrid), P_EST(1, :), P_EST(2, :))) = reGrid(sub2ind(size(tempGrid), P_EST(1, :), P_EST(2, :)));
        [reGrid, ScalingVec] = normalisePilotAverage(CEC, P_EST, reGrid);
        P_EST(3, :) = reGrid(sub2ind(size(tempGrid), P_EST(1, :), P_EST(2, :)));
    else
        error('Only UserDefined PilotAverage is supported in this simplified version');
    end
end

function [avgGrid, scalingVec] = normalisePilotAverage(CEC, p_est, reGrid)
    nPilots = length(p_est);
    avgGrid = zeros(size(reGrid));
    scalingVec = zeros(1, size(p_est, 2));

    sc = p_est(1, :)';
    sym = p_est(2, :)';

    half_freq_window = floor(CEC.FreqWindow / 2);
    half_time_window = floor(CEC.TimeWindow / 2);

    upperSC = sc - half_freq_window;
    upperSC(upperSC < 1) = 1;
    lowerSC = sc + half_freq_window;
    lowerSC(lowerSC > size(reGrid, 1)) = size(reGrid, 1);
    leftSYM = sym - half_time_window;
    leftSYM(leftSYM < 1) = 1;
    rightSYM = sym + half_time_window;
    rightSYM(rightSYM > size(reGrid, 2)) = size(reGrid, 2);

    for n = 1:nPilots
        scalingVec(n) = sum(sum(reGrid(upperSC(n):lowerSC(n), leftSYM(n):rightSYM(n)) ~= 0));
    end
    scalingVec(scalingVec == 0) = 1;
    pind = sub2ind(size(reGrid), sc, sym);
    avgGrid(pind) = reGrid(pind)./scalingVec.';
end

function H_EST = insertPilotEstimates(H_EST, P_EST, pilotIndices)
    H_EST(sub2ind(size(H_EST), pilotIndices(1, :), pilotIndices(2, :))) = P_EST(3, :);
end

function H_EST = interpolateChannel(CEC, H_EST, P_EST, pilotIndices)
    % 生成网格坐标
    [X, Y] = meshgrid(1:size(H_EST, 2), 1:size(H_EST, 1));
    Xi = Y;
    Yi = X;
    % 根据配置的插值类型进行二维插值
    H_EST = griddata(pilotIndices(2, :), pilotIndices(1, :), P_EST(3, :), Yi, Xi, CEC.InterpType);
end

function pilotIndices = generatePilotIndices(M, N)
    % 简单的导频模式，每隔4个子载波和4个OFDM符号放置一个导频
    pilotSubcarriers = 1:4:M;
    pilotSymbols = 1:4:N;
    [pilotSubcarriersGrid, pilotSymbolsGrid] = meshgrid(pilotSubcarriers, pilotSymbols);
    pilotIndices = [pilotSubcarriersGrid(:).'; pilotSymbolsGrid(:).'];
end