% 保存批量数据到文件
load('../raw/valData.mat', ...
    'csiLSData',...
    'csiPreData',...
    'csiLabelData', ...
    'txSignalData',...
    'rxSignalData');


disp(csiLabelData(1,:,1,1,1,:))