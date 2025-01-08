function save_dataset(filePath, txSignals, rxSignals, channelMatrices)
    % 保存数据集
    % 输入:
    % - filePath: 数据集保存路径
    % - txSignals, rxSignals, channelMatrices: 数据集内容

    save(filePath, 'txSignals', 'rxSignals', 'channelMatrices', '-v7.3');
end
