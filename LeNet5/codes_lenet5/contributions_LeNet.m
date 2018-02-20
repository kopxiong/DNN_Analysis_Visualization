function [matrix_contribs, input_image, input_region, position] = contributions_LeNet(images, label_index, K, kernel_level, node_level)

% tic;

% add this package to Matlab search PATH by starting matlab from caffe root folder
addpath ./matlab
addpath ../../DNN_analysis_project/LeNet5/codes_lenet5

model = '../../DNN_analysis_project/LeNet5/models_lenet5/lenet_deploy.prototxt';
weights = '../../DNN_analysis_project/LeNet5/models_lenet5/lenet_iter_20000.caffemodel';
%model = './examples/mnist/lenet_deploy.prototxt';
%weights = './examples/mnist/lenet_iter_20000.caffemodel';

caffe.set_mode_gpu();
caffe.set_device(0);

% caffe.set_mode_cpu();

% create net and load weights
net = caffe.Net(model, weights, 'test');

% % load the images and labels
% images = loadMNISTImages('train-images-idx3-ubyte');
% labels = loadMNISTLabels('train-labels-idx1-ubyte');
% 
% % % pick a small number for debug only
% % ind = floor(linspace(1, numel(labels), 1000));
% % images = images(:, ind);
% % labels = labels(ind);
%  
% label_0 = find(labels == 0);
% label_1 = find(labels == 1);
% label_2 = find(labels == 2);
% label_3 = find(labels == 3);
% label_4 = find(labels == 4);
% label_5 = find(labels == 5);
% label_6 = find(labels == 6);
% label_7 = find(labels == 7);
% label_8 = find(labels == 8);
% label_9 = find(labels == 9);

% get the weights
weights_conv1 = net.params('conv1', 1).get_data();       % size = 5 * 5 * 1 * 20
weights_conv2 = net.params('conv2', 1).get_data();       % size = 5 * 5 * 20 * 50
weights_ip1 = net.params('ip1', 1).get_data();           % size = 800 * 500
weights_ip2 = net.params('ip2', 1).get_data();           % size = 500 * 10   

% % get the bias
% bias_conv1 = net.layers('conv1').params(2).get_data();
% bias_conv2 = net.layers('conv2').params(2).get_data();
% bias_ip1 = net.layers('ip1').params(2).get_data();      
% bias_ip2 = net.layers('ip2').params(2).get_data();  

% preallocate the matrices for contributions (kernel level)
% contribs_IP1 = zeros([size(weights_conv2, 4), size(weights_ip1, 2), 1]);
% contribs_IP2 = zeros([size(weights_ip2, 1), size(weights_ip2, 2), 1]);
% contribs_CONV2 = zeros([size(weights_conv1, 4), size(weights_conv2, 4), 1]);
contribs_CONV1 = zeros([24, 24, 1]);
input_image = zeros([28, 28, 1]);
input_region = zeros([5, 5, 1]);
position = zeros([1, 2]);

% some parameters setting
num_size = length(label_index);
% batch_size = 64;
% num_iter = fix(num_size / batch_size);
matrix_contribs = zeros([num_size, 9, K]);

% visualization based on node level
if node_level == true
    for i = 1: num_size 

        display(i);
        img = images(:, label_index(i));
        input_image(:, :, i) = reshape(img, [28 28 1 1]);
        input_data = reshape(img, [28 28 1 1]);

        % transpose the data before plugging into the networKk
        res = net.forward({input_data'});
        prob = res{1};

        % get the activation values from blobs
        activation_conv1 = net.blobs('conv1').get_data();    % size = 24 * 24 * 20
        activation_pool1 = net.blobs('pool1').get_data();    % size = 12 * 12 * 20
        activation_conv2 = net.blobs('conv2').get_data();    % size = 8 * 8 * 50
        activation_pool2 = net.blobs('pool2').get_data();    % size = 4 * 4 * 50
        activation_ip1 = net.blobs('ip1').get_data();        % size = 500 * 1

        % get the maximal K contributions and indices in layer ip2
        [maxValueProb, linearIndexOfMaxProb] = max(prob(:));
        [rowsOfMaxIp2, colsOfMaxIp2] = find(prob == maxValueProb);

    %     if (rowsOfMaxIp2 - 1) ~= 6
    %         disp(sprintf('The misclassified index is: %d', i));
    %     end

        contribs_ip2_temp = weights_ip2(:, rowsOfMaxIp2).*activation_ip1;
        [sortedValuesIp2, sortedIndicesIp2] = sort(contribs_ip2_temp, 'descend');
        K_maxValuesIp2 = sortedValuesIp2(1: K);
        K_maxIndicesIp2 = sortedIndicesIp2(1: K);    

        % get the maximal K contributions and indices in layer ip1 
        [maxValueIp2, linearIndexOfMaxIp2] = max(contribs_ip2_temp(:));
        [rowsOfMaxIp1, colsOfMaxIp1] = find(contribs_ip2_temp == maxValueIp2);

        contribs_ip1_temp = weights_ip1(:, rowsOfMaxIp1).*reshape(activation_pool2, [800, 1]);  
        [sortedValuesIp1, sortedIndicesIp1] = sort(contribs_ip1_temp, 'descend');
        K_maxValuesIp1 = sortedValuesIp1(1: K);
    %     K_maxIndicesIp1 = sortedIndicesIp1(1: K); 
        K_maxIndicesIp1 = fix((sortedIndicesIp1(1: K) - 1) / 16) + 1;   

        % get the maximal K contributions and indices in blobs pool2
        [maxValueIp1, linearIndexOfMaxIp1] = max(contribs_ip1_temp(:));
        [rowsOfMaxPool2, colsOfMaxPool2] = find(contribs_ip1_temp == maxValueIp1); 

        indexOfMaxPool2 = fix((linearIndexOfMaxIp1 - 1) / 16) + 1;           % Z dimension
        heightOfMaxPool2 = fix(mod(linearIndexOfMaxIp1 - 1, 16) / 4) + 1;    % Y dimension
        widthOfMaxPool2 = mod(mod(linearIndexOfMaxIp1 - 1, 16), 4) + 1;      % X dimension

        % get the maximal K contributions and indices in blobs conv2
        maxCell_conv2 = activation_conv2(widthOfMaxPool2*2-1: widthOfMaxPool2*2, heightOfMaxPool2*2-1: heightOfMaxPool2*2, indexOfMaxPool2);
        [maxValuePool2, linearIndexOfMaxPool2] = max(maxCell_conv2(:));
        indexOfMaxConv2 = indexOfMaxPool2;
        [widthOfMaxConv2, heightOfMaxConv2] = find(activation_conv2(:, :, indexOfMaxConv2) == maxValuePool2);

        % get the maximal K contributions and indices in layer conv2 (every 5*5 region)
        contribs_conv2_temp = reshape(activation_pool1(widthOfMaxConv2:widthOfMaxConv2+4, heightOfMaxConv2:heightOfMaxConv2+4, :),...
            [500, 1]).*reshape(weights_conv2(:, :, :, indexOfMaxConv2), [500, 1]);
        [sortedValuesConv2, sortedIndicesConv2] = sort(contribs_conv2_temp, 'descend');
        K_maxValuesConv2 = sortedValuesConv2(1: K);
    %     K_maxIndicesConv2 = sortedIndicesConv2(1: K);  
        K_maxIndicesConv2 = fix((sortedIndicesConv2(1: K) - 1) / 25) + 1;   

        % get the maximal K contributions and indices in blobs pool1
        [maxValueConv2, linearIndexOfMaxConv2] = max(contribs_conv2_temp(:));
        [rowsOfMaxPool1, colsOfMaxPool1] = find(contribs_conv2_temp == maxValueConv2); 

        indexOfMaxPool1 = fix((linearIndexOfMaxConv2 - 1) / 25) + 1;
        heightOfMaxPool1 = heightOfMaxConv2 + fix(mod(linearIndexOfMaxConv2 - 1, 25) / 5);    
        widthOfMaxPool1 = widthOfMaxConv2 + mod(mod(linearIndexOfMaxConv2 - 1, 25), 5);
    %     heightOfMaxPool1 = heightOfMaxConv2 + fix(mod(linearIndexOfMaxConv2 - 1, 25) / 5) + 1;    %%%%%%
    %     widthOfMaxPool1 = widthOfMaxConv2 + mod(mod(linearIndexOfMaxConv2 - 1, 25), 5) + 1;

        % get the maximal K contributions and indices in blobs conv1
        maxCell_conv1 = activation_conv1(widthOfMaxPool1*2-1: widthOfMaxPool1*2, heightOfMaxPool1*2-1: heightOfMaxPool1*2, indexOfMaxPool1);
        [maxValuePool1, linearIndexOfMaxPool1] = max(maxCell_conv1(:));
        indexOfMaxConv1 = indexOfMaxPool1;
        [widthOfMaxConv1, heightOfMaxConv1] = find(activation_conv1(:, :, indexOfMaxConv1) == maxValuePool1);

        % get the maximal K contributions and indices in layer conv1 (back projection to the input image)
        contribs_conv1_temp = reshape(input_data(widthOfMaxConv1:widthOfMaxConv1+4, heightOfMaxConv1:heightOfMaxConv1+4), [25, 1]).* ...
            reshape(weights_conv1(:, :, indexOfMaxConv1), [25, 1]);

        [sortedValuesConv1, sortedIndicesConv1] = sort(contribs_conv1_temp, 'descend');  
        K_maxValuesConv1 = sortedValuesConv1(1: K);
        K_maxIndicesConv1 = sortedIndicesConv1(1: K);

        % store the important blobs and contribution values to matrix_contribs
        matrix_contribs(i, 1, :) = K_maxIndicesConv1;
        matrix_contribs(i, 2, :) = K_maxValuesConv1;
        matrix_contribs(i, 3, :) = K_maxIndicesConv2;
        matrix_contribs(i, 4, :) = K_maxValuesConv2;
        matrix_contribs(i, 5, :) = K_maxIndicesIp1;
        matrix_contribs(i, 6, :) = K_maxValuesIp1;
        matrix_contribs(i, 7, :) = K_maxIndicesIp2;
        matrix_contribs(i, 8, :) = K_maxValuesIp2;
        matrix_contribs(i, 9, :) = rowsOfMaxIp2;

        % store the area where the input image has the maximal contribution
        if mod(K_maxIndicesConv1(1), 24) == 0
            colOfImage = 24;
            rowOfImage = fix(K_maxIndicesConv1(1) / 24);
        else
            colOfImage = mod(K_maxIndicesConv1(1), 24);
            rowOfImage = fix(K_maxIndicesConv1(1) / 24) + 1;
        end

        input_region(:, :, i) = input_data(rowOfImage: rowOfImage+4, colOfImage: colOfImage+4);
        position(i, 1) = rowOfImage;
        position(i, 2) = colOfImage;
    end

% visualization based on kernel level
elseif kernel_level == true
    for i = 1: num_size 

        display(i);
        img = images(:, label_index(i));
        input_image(:, :, i) = reshape(img, [28 28 1 1]);
        input_data = reshape(img, [28 28 1 1]);

        % transpose the data before plugging into the networKk
        res = net.forward({input_data'});
        prob = res{1};

        % get the activation values from blobs
        activation_conv1 = net.blobs('conv1').get_data();    % size = 24 * 24 * 20
        activation_pool1 = net.blobs('pool1').get_data();    % size = 12 * 12 * 20
        activation_conv2 = net.blobs('conv2').get_data();    % size = 8 * 8 * 50
        activation_pool2 = net.blobs('pool2').get_data();    % size = 4 * 4 * 50
        activation_ip1 = net.blobs('ip1').get_data();        % size = 500 * 1

        % get the maximal K contributions and indices in layer ip2
        [maxValueProb, linearIndexesOfMaxProb] = max(prob(:));
        [rowsOfMaxIp2, colsOfMaxIp2] = find(prob == maxValueProb);

    %     if (rowsOfMaxIp2 - 1) ~= 6
    %         disp(sprintf('The misclassified index is: %d', i));
    %     end

        contribs_IP2(:, rowsOfMaxIp2, i) = weights_ip2(:, rowsOfMaxIp2).*activation_ip1;

        [sortedValuesIp2, sortedIndicesIp2] = sort(contribs_IP2(:, rowsOfMaxIp2, i), 'descend');
        K_maxValuesIp2 = sortedValuesIp2(1: K);
        K_maxIndicesIp2 = sortedIndicesIp2(1: K);    

        [maxValueIp2, linearIndexesOfMaxIp2] = max(contribs_IP2(:, rowsOfMaxIp2, i));
        [rowsOfMaxIp1, colsOfMaxIp1] = find(contribs_IP2(:, rowsOfMaxIp2, i) == maxValueIp2);

        % get the maximal K contributions and indices in layer ip1 
        contribs_ip1_temp = weights_ip1(:, rowsOfMaxIp1).*reshape(activation_pool2, [800, 1]);  
        contribs_ip1 = sum(reshape(contribs_ip1_temp, [16, 50]), 1);
        contribs_IP1(:, rowsOfMaxIp1, i) = contribs_ip1';

        [sortedValuesIp1, sortedIndicesIp1] = sort(contribs_IP1(:, rowsOfMaxIp1, i), 'descend');
        K_maxValuesIp1 = sortedValuesIp1(1: K);
        K_maxIndicesIp1 = sortedIndicesIp1(1: K);    

        [maxValueIp1, linearIndexesOfMaxIp1] = max(contribs_IP1(:, rowsOfMaxIp1, i));
        [rowsOfMaxPool2, colsOfMaxPool2] = find(contribs_IP1(:, rowsOfMaxIp1, i) == maxValueIp1);   

        % get the maximal K contributions and indices in layer conv2
        for l = 1: 20
            contribs_conv2_sum = 0;
            for j = 1: 8
                for k = 1: 8
                    contribs_conv2_layer = reshape(activation_pool1(k:k+4, j:j+4, l), [25, 1]).*reshape(weights_conv2(:, :, l, rowsOfMaxPool2), [25, 1]);
                    contribs_conv2_sum = contribs_conv2_sum + sum(contribs_conv2_layer);
                end
            end
            contribs_CONV2(l, rowsOfMaxPool2, i) = contribs_conv2_sum;
        end

        [sortedValuesConv2, sortedIndicesConv2] = sort(contribs_CONV2(:, rowsOfMaxPool2, i), 'descend');
        K_maxValuesConv2 = sortedValuesConv2(1: K);
        K_maxIndicesConv2 = sortedIndicesConv2(1: K);    

        [maxValueConv2, linearIndexesOfMaxConv2] = max(contribs_CONV2(:, rowsOfMaxPool2, i));
        [rowsOfMaxPool1, colsOfMaxPool1] = find(contribs_CONV2(:, rowsOfMaxPool2, i) == maxValueConv2); 

        % get the maximal K contributions and indices in layer conv1 (back projection to the input image)
        for j = 1: 24
            for k = 1: 24
                contribs_conv1_layer = reshape(input_data(k:k+4, j:j+4), [25, 1]).*reshape(weights_conv1(:, :, rowsOfMaxPool1), [25, 1]);
                contribs_CONV1(j, k, i) = sum(contribs_conv1_layer);
            end
        end

        contribs_temp = contribs_CONV1(:, :, i);

        [sortedValuesConv1, sortedIndicesConv1] = sort(contribs_temp(:), 'descend');  
        K_maxValuesConv1 = sortedValuesConv1(1: K);
        K_maxIndicesConv1 = sortedIndicesConv1(1: K);

        % store the important blobs and contribution values to matrix_contribs
        matrix_contribs(i, 1, :) = K_maxIndicesConv1;
        matrix_contribs(i, 2, :) = K_maxValuesConv1;
        matrix_contribs(i, 3, :) = K_maxIndicesConv2;
        matrix_contribs(i, 4, :) = K_maxValuesConv2;
        matrix_contribs(i, 5, :) = K_maxIndicesIp1;
        matrix_contribs(i, 6, :) = K_maxValuesIp1;
        matrix_contribs(i, 7, :) = K_maxIndicesIp2;
        matrix_contribs(i, 8, :) = K_maxValuesIp2;
        matrix_contribs(i, 9, :) = rowsOfMaxIp2;

        % store the area where the input image has the maximal contribution
        if mod(K_maxIndicesConv1(1), 24) == 0
            colOfImage = 24;
            rowOfImage = fix(K_maxIndicesConv1(1) / 24);
        else
            colOfImage = mod(K_maxIndicesConv1(1), 24);
            rowOfImage = fix(K_maxIndicesConv1(1) / 24) + 1;
        end

        input_region(:, :, i) = input_data(rowOfImage: rowOfImage+4, colOfImage: colOfImage+4);
        position(i, 1) = rowOfImage;
        position(i, 2) = colOfImage;
    end
end

%toc;
end