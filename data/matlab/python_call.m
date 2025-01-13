

% % 输入数据
% input = rand(2, 3); % MATLAB 矩阵，示例为 1x3 矩阵
% disp(reshape(input,1,[]))
% 
% % 转换为 Python numpy 数组
% py_input = py.numpy.array(input);
% 
% % 调用 Python 脚本
% py_output = py.infer.test(py_input);
% 
% % 转换 Python numpy 输出为 MATLAB 矩阵
% output = double(py.array.array('d', py.numpy.nditer(py_output)));

model = py.infer.load_model();
py.infer.test(model);
% % 显示结果
% disp('Output from Python:');
% disp(output);