function [rxSignal, csi] = apply_channel(channel, txSignal)
    % 信号通过信道传播
    % 输入:
    % - channel: 信道对象
    % - txSignal: 发射信号
    % 输出:
    % - rxSignal: 接收信号
    % - csi: 信道状态信息 (CSI)
    
    [rxSignal, pathGains] = channel(txSignal);
    csi = getCSI(pathGains, channel);
end

function csi = getCSI(pathGains, channel)
    % 提取信道状态信息 (CSI)
    % 计算基带信道矩阵
    numTx = channel.NumTransmitAntennas;
    numRx = channel.NumReceiveAntennas;
    csi = reshape(mean(pathGains, 1), numRx, numTx);
end
