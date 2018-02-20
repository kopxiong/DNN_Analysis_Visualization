function [m_contribs_IP2, m_contribs_IP1, m_contribs_CONV2] = top_K_cons(labels_index, K)

% add this package to Matlab search PATH by starting matlab from caffe root folder
addpath ./matlab
addpath ../../masterproject/codes

model = './examples/mnist/lenet_yufeng.prototxt';
weights = './examples/mnist/lenet_iter_20000.caffemodel';

% caffe.set_mode_gpu();
% caffe.set_device(0);

caffe.set_mode_cpu();

% create net and load weights
net = caffe.Net(model, weights, 'test');

% load the training images and labels
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');

% % pick a small number for debug only
% ind = floor(linspace(1, numel(labels), 1000));
% images = images(:, ind);
% labels = labels(ind);

% % get the indices of different labels
% labels_0 = find(labels == 0);
% labels_1 = find(labels == 1);
% labels_2 = find(labels == 2);
% labels_3 = find(labels == 3);
% labels_4 = find(labels == 4);
% labels_5 = find(labels == 5);
% labels_6 = find(labels == 6);
% labels_7 = find(labels == 7);
% labels_8 = find(labels == 8);
% labels_9 = find(labels == 9);

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
    
% preallocate the matrices for contributions
contribs_CONV2 = zeros([size(weights_conv1, 4), size(weights_conv2, 4), 1]);    
contribs_IP1 = zeros([size(weights_conv2, 4), size(weights_ip1, 2), 1]);
contribs_IP2 = zeros([size(weights_ip1, 2), size(weights_ip2, 2), 1]);

m_contribs_CONV2 = zeros([size(weights_conv1, 4), size(weights_conv2, 4)]);
m_contribs_IP1 = zeros([size(weights_conv2, 4), size(weights_ip1, 2)]);
m_contribs_IP2 = zeros([size(weights_ip1, 2), size(weights_ip2, 2)]);

% % to track the maximal K contributions
% K_max_contribsConv2 = zeros(length(labels_index), K);
% K_max_contribsIp1 = zeros(length(labels_index), K);
% K_max_contribsIp2 = zeros(length(labels_index), K);
% 
% K_node_pool1 = zeros(length(labels_index), K);
% K_node_pool2 = zeros(length(labels_index), K);
% K_node_ip1 = zeros(length(labels_index), K);

for i = 1: length(labels_index) 
    display(i);
    img = images(:, labels_index(i));
    data = reshape(img, [28 28 1 1]);

    % transpose the data before plugging into the network
    res = net.forward({data'});
    prob = res{1};

    % get the activation values from blobs
    activation_pool1 = net.blobs('pool1').get_data();    % size = 12 * 12 * 20
    activation_pool2 = net.blobs('pool2').get_data();    % size = 4 * 4 * 50
    activation_ip1 = net.blobs('ip1').get_data();        % size = 500 * 1
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % keep top K contributions of layer ip2 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [maxValueProb, linearIndexesOfMaxProb] = max(prob(:));
    [rowsOfMaxIp2, colsOfMaxIp2] = find(prob == maxValueProb);

    contribs_IP2(:, rowsOfMaxIp2, i) = weights_ip2(:, rowsOfMaxIp2).*activation_ip1;

    [sortedValuesIp2, sortedIndicesIp2] = sort(contribs_IP2(:, rowsOfMaxIp2, i), 'descend');
    K_maxValuesIp2 = sortedValuesIp2(1: K);
    K_maxIndicesIp2 = sortedIndicesIp2(1: K);    

    m_contribs_IP2(K_maxIndicesIp2, rowsOfMaxIp2) = m_contribs_IP2(K_maxIndicesIp2, rowsOfMaxIp2) + K_maxValuesIp2;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % keep top K contributions of layer ip2 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    [maxValueIp2, linearIndexesOfMaxIp2] = max(contribs_IP2(:, rowsOfMaxIp2, i));
    [rowsOfMaxIp1, colsOfMaxIp1] = find(contribs_IP2(:, rowsOfMaxIp2, i) == maxValueIp2);
    
    contribs_ip1_temp = weights_ip1(:, rowsOfMaxIp1).*reshape(activation_pool2, [800, 1]);  
    contribs_ip1 = sum(reshape(contribs_ip1_temp, [16, 50]), 1);
    contribs_IP1(:, rowsOfMaxIp1, i) = contribs_ip1';
                      
    [sortedValuesIp1, sortedIndicesIp1] = sort(contribs_IP1(:, rowsOfMaxIp1, i), 'descend');
    K_maxValuesIp1 = sortedValuesIp1(1: K);
    K_maxIndicesIp1 = sortedIndicesIp1(1: K);    
    
    m_contribs_IP1(K_maxIndicesIp1, rowsOfMaxIp1) = m_contribs_IP1(K_maxIndicesIp1, rowsOfMaxIp1) + K_maxValuesIp1;
 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % keep top K contributions of layer conv2 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [maxValueIp1, linearIndexesOfMaxIp1] = max(contribs_IP1(:, rowsOfMaxIp1, i));
    [rowsOfMaxPool2, colsOfMaxPool2] = find(contribs_IP1(:, rowsOfMaxIp1, i) == maxValueIp1);   

    for l = 1: 20
        contribs_conv2_sum = 0;
        for j = 1: 8
            for k = 1: 8
                contribs_conv2_layer = reshape(activation_pool1(k:k+4, j:j+4, l), [25, 1]).*reshape(weights_conv2(:, :, l, rowsOfMaxPool2), [25, 1]);
                contribs_conv2_sum = contribs_conv2_sum + sum(contribs_conv2_layer);
            end
        end
        m_contribs_CONV2(l, rowsOfMaxPool2) = m_contribs_CONV2(l, rowsOfMaxPool2) + contribs_conv2_sum;
    end
end

end