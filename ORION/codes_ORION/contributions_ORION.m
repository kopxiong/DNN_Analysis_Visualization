function matrix_contribs = contributions_ORION(mn10_data, ORION_version, label_index, K, kernel_level, node_level)

% add this package to Matlab search PATH by starting matlab from caffe root folder
addpath ./matlab
addpath ../../DNN_analysis_project/ORION/codes_ORION

% Baseline model and weights
if strcmp(ORION_version, 'Baseline')
    model = '../../DNN_analysis_project/ORION/baseline/deploy_baseline.prototxt';
    weights = '../../DNN_analysis_project/ORION/baseline/snapshot_iter_50000.caffemodel';

% Basic model and weights
elseif strcmp(ORION_version, 'Basic')
    model = '../../DNN_analysis_project/ORION/basic/deploy_basic.prototxt';
    weights = '../../DNN_analysis_project/ORION/basic/snapshot_iter_500000.caffemodel';
end

caffe.set_mode_gpu();
caffe.set_device(0);

% caffe.set_mode_cpu();

% create net and load weights
net = caffe.Net(model, weights, 'test');

% get the weights
weights_conv1a = net.params('conv1a', 1).get_data();       % size = 5 * 5 * 5 * 1 * 32
weights_conv2a = net.params('conv2a', 1).get_data();       % size = 3 * 3 * 3 * 32 * 32
weights_fc6 = net.params('fc6', 1).get_data();             % size = 6912 * 128
weights_fc8 = net.params('fc8', 1).get_data();             % size = 128 * 10   
% weights_fc8_pose = net.params('fc8_pose', 1).get_data();   % size = 128 * 105   

% get the bias
bias_conv1a = net.layers('conv1a').params(2).get_data();
bias_conv2a = net.layers('conv2a').params(2).get_data();
bias_fc6 = net.layers('fc6').params(2).get_data();      
bias_fc8 = net.layers('fc8').params(2).get_data();  

% % preallocate the matrices for contributions (kernel level)
% contribs_CONV1A = zeros([14, 14, 14, 1]);
% contribs_CONV2A = zeros([size(weights_conv1a, 5), size(weights_conv2a, 5), 1]);
% contribs_FC6 = zeros([size(weights_conv2a, 5), size(weights_fc6, 2), 1]);
% contribs_FC8 = zeros([size(weights_fc8, 1), size(weights_fc8, 2), 1]);

% number of iterations and store all important nodes and contributions into matrix
num_size = length(label_index);
% batch_size = 32;
matrix_contribs = zeros([num_size, 9, K]);
% num_iter = fix(num_size / batch_size); 
% reminder = mod(num_size, batch_size);
% counter = 0;

% visualization based on kernel level
if kernel_level == true
    for i = 1: num_size 
        display(i);
        batch_data = mn10_data(:, :, :, :, label_index(i));

        % plug input data into the network (here do not need permute)
        res = net.forward({batch_data});
        prob = res{1};

        % get the activation values from blobs   
        activation_data = net.blobs('data').get_data();                         % size = 36 * 36 * 36 * 1 * 32
        activation_def = net.blobs('def').get_data();                           % size = 3 * 32 * 32 * 32 * 32
        activation_data_aug = net.blobs('data_aug').get_data();                 % size = 32 * 32 * 32 * 1 * 32
        activation_conv1a = net.blobs('conv1a').get_data();                     % size = 14 * 14 * 14 * 32 * 32
        activation_conv2a = net.blobs('conv2a').get_data();                     % size = 12 * 12 * 12 * 32 * 32
        activation_pool2a = net.blobs('pool2a').get_data();                     % size = 6 * 6 * 6 * 32 * 32
        activation_fc6 = net.blobs('fc6').get_data();                           % size = 128 * 32
    %     activation_fc6_0 = net.blobs('fc6_relu6_0_split_0').get_data();         % size = 128 * 32
    %     activation_fc6_1 = net.blobs('fc6_relu6_0_split_1').get_data();         % size = 128 * 32
    %     activation_fc8 = net.blobs('fc8').get_data();                           % size = 10 * 32
    %     activation_fc8_pose = net.blobs('fc8_pose').get_data();                 % size = 105 * 32
    %     activation_prob_label = net.blobs('prob_label').get_data();             % size = 10 * 32
    %     activation_prob_pose = net.blobs('prob_pose').get_data();               % size = 105 * 32

        % get the maximal contributions and indices in layer fc8
        [maxValueProb, linearIndexesOfMaxProb] = max(prob);
        [rowsOfMaxFc8, colsOfMaxFc8] = find(prob == maxValueProb);

    %     if (rowsOfMaxFc8 - 1) ~= mn10_label(i)
    %         disp(sprintf('The misclassified index is: %d', i));
    %         counter = counter + 1;
    %     end

        contribs_FC8(:, rowsOfMaxFc8, i) = weights_fc8(:, rowsOfMaxFc8).*activation_fc6;    %%%%%%%

        [sortedValuesFc8, sortedIndicesFc8] = sort(contribs_FC8(:, rowsOfMaxFc8, i), 'descend');
        K_maxValuesFc8 = sortedValuesFc8(1: K);
        K_maxIndicesFc8 = sortedIndicesFc8(1: K);    

        [maxValueFc8, linearIndexesOfMaxFc8] = max(contribs_FC8(:, rowsOfMaxFc8, i));
        [rowsOfMaxFc6, colsOfMaxFc6] = find(contribs_FC8(:, rowsOfMaxFc8, i) == maxValueFc8);

        % get the maximal K contributions and indices in layer fc6 
        contribs_fc6_temp = weights_fc6(:, rowsOfMaxFc6).*reshape(activation_pool2a, [6912, 1]);  
        contribs_fc6 = sum(reshape(contribs_fc6_temp, [216, 32]), 1);
        contribs_FC6(:, rowsOfMaxFc6, i) = contribs_fc6';

        [sortedValuesFc6, sortedIndicesFc6] = sort(contribs_FC6(:, rowsOfMaxFc6, i), 'descend');
        K_maxValuesFc6 = sortedValuesFc6(1: K);
        K_maxIndicesFc6 = sortedIndicesFc6(1: K);    

        [maxValueFc6, linearIndexesOfMaxFc6] = max(contribs_FC6(:, rowsOfMaxFc6, i));
        [rowsOfMaxPool2a, colsOfMaxPool2a] = find(contribs_FC6(:, rowsOfMaxFc6, i) == maxValueFc6);   

        % get the maximal K contributions and indices in layer conv2a
        for j = 1: 32
            contribs_conv2a_sum = 0;
            for k = 1: 12
                for l = 1: 12
                    for m = 1: 12
                        contribs_conv2a_layer = reshape(activation_conv1a(m:m+2, l:l+2, k:k+2, j), [27, 1]).*reshape(weights_conv2a(:, :, :, j, rowsOfMaxPool2a), [27, 1]);
                        contribs_conv2a_sum = contribs_conv2a_sum + sum(contribs_conv2a_layer);
                    end
                end
            end
            contribs_CONV2A(j, rowsOfMaxPool2a, i) = contribs_conv2a_sum;
        end   

        [sortedValuesConv2a, sortedIndicesConv2a] = sort(contribs_CONV2A(:, rowsOfMaxPool2a, i), 'descend');
        K_maxValuesConv2a = sortedValuesConv2a(1: K);
        K_maxIndicesConv2a = sortedIndicesConv2a(1: K);    

        [maxValueConv2a, linearIndexesOfMaxConv2a] = max(contribs_CONV2A(:, rowsOfMaxPool2a, i));
        [rowsOfMaxConv1a, colsOfMaxConv1a] = find(contribs_CONV2A(:, rowsOfMaxPool2a, i) == maxValueConv2a); 

        % get the maximal K contributions and indices in layer conv1a (back projection to the input image)
        for k = 1: 14
            for l = 1: 14
                for m = 1: 14
                    contribs_conv1a_layer = reshape(batch_data(m:m+4, l:l+4, k:k+4), [125, 1]).*reshape(weights_conv1a(:, :, :, :, rowsOfMaxConv1a), [125, 1]);
                    contribs_CONV1A(k, l, m, i) = sum(contribs_conv1a_layer);
                end
            end
        end

        contribs_temp = contribs_CONV1A(:, :, :, i);

        [sortedValuesConv1a, sortedIndicesConv1a] = sort(contribs_temp(:), 'descend');  
        K_maxValuesConv1a = sortedValuesConv1a(1: K);
        K_maxIndicesConv1a = sortedIndicesConv1a(1: K);

        % store the important blobs and contribution values to matrix_contribs
        matrix_contribs(i, 1, :) = K_maxIndicesConv1a;
        matrix_contribs(i, 2, :) = K_maxValuesConv1a;
        matrix_contribs(i, 3, :) = K_maxIndicesConv2a;
        matrix_contribs(i, 4, :) = K_maxValuesConv2a;
        matrix_contribs(i, 5, :) = K_maxIndicesFc6;
        matrix_contribs(i, 6, :) = K_maxValuesFc6;
        matrix_contribs(i, 7, :) = K_maxIndicesFc8;
        matrix_contribs(i, 8, :) = K_maxValuesFc8;
        matrix_contribs(i, 9, :) = rowsOfMaxFc8;

    end

% visualization based on node level
elseif node_level == true
    for i = 1: num_size 
        display(i);
        batch_data = mn10_data(:, :, :, :, label_index(i));

        % plug input data into the network (hereposition do not need permute)
        res = net.forward({batch_data});
        prob = res{1};

        % get the activation values from blobs   
        activation_data = net.blobs('data').get_data();                         % size = 36 * 36 * 36 * 1 * 32
        activation_def = net.blobs('def').get_data();                           % size = 3 * 32 * 32 * 32 * 32
        activation_data_aug = net.blobs('data_aug').get_data();                 % size = 32 * 32 * 32 * 1 * 32
        activation_conv1a = net.blobs('conv1a').get_data();                     % size = 14 * 14 * 14 * 32 * 32
        activation_conv2a = net.blobs('conv2a').get_data();                     % size = 12 * 12 * 12 * 32 * 32
        activation_pool2a = net.blobs('pool2a').get_data();                     % size = 6 * 6 * 6 * 32 * 32
        activation_fc6 = net.blobs('fc6').get_data();                           % size = 128 * 32
    %     activation_fc6_0 = net.blobs('fc6_relu6_0_split_0').get_data();         % size = 128 * 32
    %     activation_fc6_1 = net.blobs('fc6_relu6_0_split_1').get_data();         % size = 128 * 32
    %     activation_fc8 = net.blobs('fc8').get_data();                           % size = 10 * 32
    %     activation_fc8_pose = net.blobs('fc8_pose').get_data();                 % size = 105 * 32
    %     activation_prob_label = net.blobs('prob_label').get_data();             % size = 10 * 32
    %     activation_prob_pose = net.blobs('prob_pose').get_data();               % size = 105 * 32

        % 1.get the maximal contributions and indices in layer fc8
        [maxValueProb, linearIndexOfMaxProb] = max(prob);
        [rowsOfMaxFc8, colsOfMaxFc8] = find(prob == maxValueProb);

    %     % print the number of misclassifed examples
    %     if (rowsOfMaxFc8 - 1) ~= mn10_label(i)
    %         disp(sprintf('The misclassified index is: %d', i));
    %         counter = counter + 1;
    %     end

        contribs_fc8_temp = weights_fc8(:, rowsOfMaxFc8).*activation_fc6;        
        [sortedValuesFc6, sortedIndicesFc6] = sort(contribs_fc8_temp, 'descend');
        K_maxValuesFc6 = sortedValuesFc6(1: K);
        K_maxIndicesFc6 = sortedIndicesFc6(1: K);    

        % 2.get the maximal K contributions and indices in layer fc6 
        [maxValueFc8, linearIndexOfMaxFc8] = max(contribs_fc8_temp(:));
        [rowsOfMaxFc6, colsOfMaxFc6] = find(contribs_fc8_temp == maxValueFc8);

        contribs_fc6_temp = weights_fc6(:, rowsOfMaxFc6).*reshape(activation_pool2a, [6912, 1]);  
        [sortedValuesPool2a, sortedIndicesPool2a] = sort(contribs_fc6_temp, 'descend');    
        K_maxValuesPool2a = sortedValuesPool2a(1: K);
        K_maxIndicesPool2a = fix((sortedIndicesPool2a(1: K) - 1) / 216) + 1;    

        % 3.get the maximal K contributions and indices in blobs pool2a
        [maxValueFc6, linearIndexOfMaxFc6] = max(contribs_fc6_temp(:));
        [rowsOfMaxPool2a, colsOfMaxPool2a] = find(contribs_fc6_temp == maxValueFc6);  

        indexOfMaxPool2a = fix((linearIndexOfMaxFc6 - 1) / 216) + 1;          % keep the index of the most important filter
        depthOfMaxPool2a = fix(mod(linearIndexOfMaxFc6 - 1, 216) / 36) + 1;   % Z dimension
        heightOfMaxPool2a = fix(mod(linearIndexOfMaxFc6 - 1, 36) / 6) + 1;    % Y dimension
        widthOfMaxPool2a = mod(mod(linearIndexOfMaxFc6 - 1, 36), 6) + 1;      % X dimension

        % get the maximal K contributions and indices in blobs conv2a
        maxVolume = activation_conv2a(widthOfMaxPool2a*2-1: widthOfMaxPool2a*2, heightOfMaxPool2a*2-1: heightOfMaxPool2a*2, ...
            depthOfMaxPool2a*2-1: depthOfMaxPool2a*2, indexOfMaxPool2a);
        [maxValuePool2a, linearIndexOfMaxPool2a] = max(maxVolume(:));
        indexOfMaxConv2a = indexOfMaxPool2a;
        [widthOfMaxConv2a, heightOfMaxConv2a, depthOfMaxConv2a] = ind2sub(size(activation_conv2a(:, :, :, indexOfMaxConv2a)), ...
            find(activation_conv2a(:, :, :, indexOfMaxConv2a) == maxValuePool2a));
        K_maxValuesConv2a = K_maxValuesPool2a;
        K_maxIndicesConv2a = K_maxIndicesPool2a;

        % get the maximal K contributions and indices in layer conv2a (Be careful: here used leaky ReLUs!)
        contribs_conv2a_temp = reshape(activation_conv1a(widthOfMaxConv2a:widthOfMaxConv2a+2, heightOfMaxConv2a:heightOfMaxConv2a+2, ...
            depthOfMaxConv2a:depthOfMaxConv2a+2, :), [864, 1]).*reshape(weights_conv2a(:, :, :, :, indexOfMaxConv2a), [864, 1]);    % 3*3*3*32   
        [sortedValuesConv1a, sortedIndicesConv1a] = sort(contribs_conv2a_temp, 'descend');
        K_maxValuesConv1a = sortedValuesConv1a(1: K);
        K_maxIndicesConv1a = fix((sortedIndicesConv1a(1: K) - 1) / 27) + 1;

        [maxValueConv2a, linearIndexOfMaxConv2a] = max(contribs_conv2a_temp(:));
        [rowsOfMaxConv1a, colsOfMaxConv1a] = find(contribs_conv2a_temp == maxValueConv2a);  

        indexOfMaxConv1a = fix((linearIndexOfMaxConv2a - 1) / 27) + 1;                             % keep the index of the most important filter
        depthOfMaxConv1a = depthOfMaxConv2a + fix(mod(linearIndexOfMaxConv2a - 1, 27) / 9);    % Z dimension
        heightOfMaxConv1a = heightOfMaxConv2a + fix(mod(linearIndexOfMaxConv2a - 1, 9) / 3);   % Y dimension
        widthOfMaxConv1a = widthOfMaxConv2a + mod(mod(linearIndexOfMaxConv2a - 1, 9), 3);      % X dimension

    %     depthOfMaxConv1a = depthOfMaxConv2a + fix(mod(linearIndexOfMaxConv2a - 1, 27) / 9) + 1;    % Z dimension
    %     heightOfMaxConv1a = heightOfMaxConv2a + fix(mod(linearIndexOfMaxConv2a - 1, 9) / 3) + 1;   % Y dimension
    %     widthOfMaxConv1a = widthOfMaxConv2a + mod(mod(linearIndexOfMaxConv2a - 1, 9), 3) + 1;      % X dimension


        % get the maximal K contributions and indices in layer conv1a (back projection to the input image)
        contribs_conv1a_temp = reshape(batch_data(widthOfMaxConv1a:widthOfMaxConv1a+4, heightOfMaxConv1a:heightOfMaxConv1a+4, ...
            depthOfMaxConv1a:depthOfMaxConv1a+4, :), [125, 1]).*reshape(weights_conv1a(:, :, :, :, indexOfMaxConv1a), [125, 1]);    % 5*5*5*1   
        [sortedValuesInput, sortedIndicesInput] = sort(contribs_conv1a_temp, 'descend');
        K_maxValuesInput = sortedValuesInput(1, K);
        K_maxIndicesInput = sortedIndicesInput(1, K);

    %     K_maxValuesInput = ones(K, 1);
    %     K_maxIndicesInput = repmat(indexOfMaxConv1a, [K, 1]);

        % store the important blobs and contribution values to matrix_contribs
        matrix_contribs(i, 1, :) = K_maxIndicesInput;
        matrix_contribs(i, 2, :) = K_maxValuesInput;
        matrix_contribs(i, 3, :) = K_maxIndicesConv1a;
        matrix_contribs(i, 4, :) = K_maxValuesConv1a;
        matrix_contribs(i, 5, :) = K_maxIndicesConv2a;
        matrix_contribs(i, 6, :) = K_maxValuesConv2a;
        matrix_contribs(i, 7, :) = K_maxIndicesFc6;
        matrix_contribs(i, 8, :) = K_maxValuesFc6;
        matrix_contribs(i, 9, :) = rowsOfMaxFc8;

    end
end
end