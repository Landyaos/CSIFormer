test = reshape(1:(2*3*4*5), [2, 3, 4, 5]);
disp(size([unique(test);1]))
[unique(test);1]