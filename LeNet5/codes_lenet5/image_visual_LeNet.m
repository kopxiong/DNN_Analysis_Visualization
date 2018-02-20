function image_visual_LeNet(images, label_index, K, count_flag, kernel_level, node_level, sum_norm_flag, max_norm_flag, renorm_flag)

% initialize the network with 620 * 490 pixel (kenel level)
M = zeros([620, 490]);
M(:, 1: 9) = 0.1;
M(:, 121: 129) = 0.1;
M(:, 241: 249) = 0.1;
M(:, 361: 369) = 0.1;
M(:, 481: 490) = 0.1;
M((80: 60: 560), 481: 490) = 0.2;
M((32: 12: 608), 241: 249) = 0.2;
M((50: 30: 590), 121: 129) = 0.2;

% initialize the layer-link matrices
m_link_CONV1 = zeros([620, 490]);
m_link_CONV2 = zeros([620, 490]);
m_link_IP1 = zeros([620, 490]);
m_link_IP2 = zeros([620, 490]);

% call function contributions to get the matrix_contribs
[mat_contribs, input_image, input_region, position] = contributions_LeNet(images, label_index, K, kernel_level, node_level);

% concatenate the mat_contribs, here the links based on the contribution values
matrix_contribs = [];
for i = 1: K
    matrix_contribs = [matrix_contribs; mat_contribs(:, :, i)];
end

% Visualization based on value or frequency
if count_flag == true
    matrix_contribs(:, 2: 2: end) = 1;
end

% accumulate the contributions in layer conv1 (back-projection to the input image)
[POOL1, ~, i_pool1] = unique([matrix_contribs(:, 1), matrix_contribs(:, 3)], 'rows');
m_input_pool1 = zeros([length(POOL1(:, 1)), 3]);
m_input_pool1(:, 1) = POOL1(:, 1);
m_input_pool1(:, 3) = POOL1(:, 2);
for i = 1: length(POOL1(:, 1))
    compoundCondInd = (matrix_contribs(:, 1) == m_input_pool1(i, 1)) & (matrix_contribs(:, 3) == m_input_pool1(i, 3));
    m_input_pool1(i, 2) = sum(matrix_contribs(compoundCondInd, 2));
end

% accumulate the contributions in layer conv2
[POOL2, ~, i_pool2] = unique([matrix_contribs(:, 3), matrix_contribs(:, 5)], 'rows');
m_pool1_pool2 = zeros([length(POOL2(:, 1)), 3]);
m_pool1_pool2(:, 1) = POOL2(:, 1);
m_pool1_pool2(:, 3) = POOL2(:, 2);
for i = 1: length(POOL2(:, 1))
    compoundCondInd = (matrix_contribs(:, 3) == m_pool1_pool2(i, 1)) & (matrix_contribs(:, 5) == m_pool1_pool2(i, 3));
    m_pool1_pool2(i, 2) = sum(matrix_contribs(compoundCondInd, 4));
end

% accumulate the contributions in layer ip1
[IP1, ~, i_ip1] = unique([matrix_contribs(:, 5), matrix_contribs(:, 7)], 'rows');
m_pool2_ip1 = zeros([length(IP1(:, 1)), 3]);
m_pool2_ip1(:, 1) = IP1(:, 1);
m_pool2_ip1(:, 3) = IP1(:, 2);
for i = 1: length(IP1(:, 1))
    compoundCondInd = (matrix_contribs(:, 5) == m_pool2_ip1(i, 1)) & (matrix_contribs(:, 7) == m_pool2_ip1(i, 3));
    m_pool2_ip1(i, 2) = sum(matrix_contribs(compoundCondInd, 6));
end

% accumulate the contributions in layer ip2
[IP2, ~, i_ip2] = unique([matrix_contribs(:, 7), matrix_contribs(:, 9)], 'rows');
m_ip1_ip2 = zeros([length(IP2(:, 1)), 3]);
m_ip1_ip2(:, 1) = IP2(:, 1);
m_ip1_ip2(:, 3) = IP2(:, 2);
for i = 1: length(IP2(:, 1))
    compoundCondInd = (matrix_contribs(:, 7) == m_ip1_ip2(i, 1)) & (matrix_contribs(:, 9) == m_ip1_ip2(i, 3));
    m_ip1_ip2(i, 2) = sum(matrix_contribs(compoundCondInd, 8));
end

% normalize the contributions in layer conv2, ip1, ip2 separately (sum or max)
if sum_norm_flag == true
    m_input_pool1(:, 2) = m_input_pool1(:, 2) / sum(m_input_pool1(:, 2));
    m_pool1_pool2(:, 2) = m_pool1_pool2(:, 2) / sum(m_pool1_pool2(:, 2));
    m_pool2_ip1(:, 2) = m_pool2_ip1(:, 2) / sum(m_pool2_ip1(:, 2));
    m_ip1_ip2(:, 2) = m_ip1_ip2(:, 2) / sum(m_ip1_ip2(:, 2));

% max normalization
elseif max_norm_flag == true
    m_input_pool1(:, 2) = m_input_pool1(:, 2) / max(m_input_pool1(:, 2));
    m_pool1_pool2(:, 2) = m_pool1_pool2(:, 2) / max(m_pool1_pool2(:, 2));
    m_pool2_ip1(:, 2) = m_pool2_ip1(:, 2) / max(m_pool2_ip1(:, 2));
    m_ip1_ip2(:, 2) = m_ip1_ip2(:, 2) / max(m_ip1_ip2(:, 2));
end

% add some switch parameters to control whether to re-normalize or not (contribution propagation)
if renorm_flag == true
    unique_ip1 = unique(m_pool2_ip1(:, 3));
    unique_pool2 = unique(m_pool1_pool2(:, 3));
    unique_pool1 = unique(m_input_pool1(:, 3));

    for i = 1: length(unique_ip1)
        m_pool2_ip1(m_pool2_ip1(:, 3) == unique_ip1(i), 2) = m_pool2_ip1(m_pool2_ip1(:, 3) == unique_ip1(i), 2) ...
            * sum(m_ip1_ip2(m_ip1_ip2(:, 1) == unique_ip1(i), 2)) / sum(m_pool2_ip1(m_pool2_ip1(:, 3) == unique_ip1(i), 2));
    end

    for j = 1: length(unique_pool2)
        m_pool1_pool2(m_pool1_pool2(:, 3) == unique_pool2(j), 2) = m_pool1_pool2(m_pool1_pool2(:, 3) == unique_pool2(j), 2) ...
            * sum(m_pool2_ip1(m_pool2_ip1(:, 1) == unique_pool2(j), 2)) / sum(m_pool1_pool2(m_pool1_pool2(:, 3) == unique_pool2(j), 2));
    end

    for p = 1: length(unique_pool1)
        m_input_pool1(m_input_pool1(:, 3) == unique_pool1(p), 2) = m_input_pool1(m_input_pool1(:, 3) == unique_pool1(p), 2) ...
            * sum(m_pool1_pool2(m_pool1_pool2(:, 1) == unique_pool1(p), 2)) / sum(m_input_pool1(m_input_pool1(:, 3) == unique_pool1(p), 2));
    end
end


% find the most important node in each layer, always keep the first largest node
[unique_ip2, ~, index_ip2] = unique(m_ip1_ip2(:, 3));
accum_ip2 = [unique_ip2, accumarray(index_ip2, m_ip1_ip2(:, 2))];
max_index_ip2 = find(accum_ip2(:, 2) == max(accum_ip2(:, 2)));
max_ip2 = accum_ip2(max_index_ip2(1));

[unique_ip1, ~, index_ip1] = unique(m_pool2_ip1(:, 3));
accum_ip1 = [unique_ip1, accumarray(index_ip1, m_pool2_ip1(:, 2))];
max_index_ip1 = find(accum_ip1(:, 2) == max(accum_ip1(:, 2)));
max_ip1 = accum_ip1(max_index_ip1(1));

[unique_pool2, ~, index_pool2] = unique(m_pool1_pool2(:, 3));
accum_pool2 = [unique_pool2, accumarray(index_pool2, m_pool1_pool2(:, 2))];
max_index_pool2 = find(accum_pool2(:, 2) == max(accum_pool2(:, 2)));
max_pool2 = accum_pool2(max_index_pool2(1));    % what if two nodes have the same value?

[unique_pool1, ~, index_pool1] = unique(m_input_pool1(:, 3));
accum_pool1 = [unique_pool1, accumarray(index_pool1, m_input_pool1(:, 2))];
max_index_pool1 = find(accum_pool1(:, 2) == max(accum_pool1(:, 2)));
max_pool1 = accum_pool1(max_index_pool1(1));


% get the link for layer ip2
for u = 1: length(m_ip1_ip2(:, 1))   
    [rows_link2, cols_link2] = bresenham(m_ip1_ip2(u, 1)+70, 371, (m_ip1_ip2(u, 3)-1)*60+50, 480);
    for m = 1: length(rows_link2)
        m_link_IP2(rows_link2(m), cols_link2(m)) = m_link_IP2(rows_link2(m), cols_link2(m)) + m_ip1_ip2(u, 2);
    end
end

% get the link for layer ip1
for v = 1: length(m_pool2_ip1(:, 1))
    [rows_link1, cols_link1] = bresenham((m_pool2_ip1(v, 1)-1)*12+26, 251, m_pool2_ip1(v, 3)+70, 360);
    for m = 1: length(rows_link1)
        m_link_IP1(rows_link1(m), cols_link1(m)) = m_link_IP1(rows_link1(m), cols_link1(m)) + m_pool2_ip1(v, 2);
    end
end

% get the link for layer conv2
for w = 1: length(m_pool1_pool2(:, 1))
    [rows_conv2, cols_conv2] = bresenham((m_pool1_pool2(w, 1)-1)*30+35, 131, (m_pool1_pool2(w, 3)-1)*12+26, 240);
    for m = 1: length(rows_conv2)
        m_link_CONV2(rows_conv2(m), cols_conv2(m)) = m_link_CONV2(rows_conv2(m), cols_conv2(m)) + m_pool1_pool2(w, 2);
    end
end

% % get the link for layer conv1 (back-projection to the input image)
% for x = 1: length(m_input_pool1(:, 1))
%     [rows_conv1, cols_conv1] = bresenham(m_input_pool1(x, 1)+32, 11, (m_input_pool1(x, 3)-1)*30+35, 120);
%     for m = 1: length(rows_conv1)
%         m_link_CONV1(rows_conv1(m), cols_conv1(m)) = m_link_CONV1(rows_conv1(m), cols_conv1(m)) + m_input_pool1(x, 2);
%     end
% end

% M = M + m_link_IP2 + m_link_IP1 + m_link_CONV2 + m_link_CONV1;
M = M + m_link_IP2 + m_link_IP1 + m_link_CONV2;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Write significant contributions in the image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1. Collect the significant contributions value (like larger than 0.1)
contribs_threshold = 0.1;
value_ip1_ip2 = m_ip1_ip2(m_ip1_ip2(:, 2) > contribs_threshold, 2);
value_pool2_ip1 = m_pool2_ip1(m_pool2_ip1(:, 2) > contribs_threshold, 2);
value_pool1_pool2 = m_pool1_pool2(m_pool1_pool2(:, 2) > contribs_threshold, 2);
% value_input_pool1 = m_input_pool1(m_input_pool1(:, 2) > contribs_threshold, 2);
% value = cat(1, value_ip1_ip2, value_pool2_ip1, value_pool1_pool2, value_input_pool1);
% value = cat(1, value_ip1_ip2, value_pool2_ip1, value_pool1_pool2);

% concatenate the value of important nodes
max_blobs = cat(1, max_ip2, max_ip1, max_pool2, max_pool1);
value = cat(1, value_ip1_ip2, value_pool2_ip1, value_pool1_pool2, max_blobs);

% 2. Collect the corresponding positions
indices_ip1_ip2 = [m_ip1_ip2(m_ip1_ip2(:, 2) > contribs_threshold, 1), m_ip1_ip2(m_ip1_ip2(:, 2) > contribs_threshold, 3)];
positions_ip1_ip2 = [repmat(425, length(indices_ip1_ip2(:, 1)), 1), mean([indices_ip1_ip2(:, 1)+70, (indices_ip1_ip2(:, 2)-1)*60+50], 2)];

indices_pool2_ip1 = [m_pool2_ip1(m_pool2_ip1(:, 2) > contribs_threshold, 1), m_pool2_ip1(m_pool2_ip1(:, 2) > contribs_threshold, 3)];
positions_pool2_ip1 = [repmat(305, length(indices_pool2_ip1(:, 1)), 1), mean([(indices_pool2_ip1(:, 1)-1)*12+26, indices_pool2_ip1(:, 2)+70], 2)];

indices_pool1_pool2 = [m_pool1_pool2(m_pool1_pool2(:, 2) > contribs_threshold, 1), m_pool1_pool2(m_pool1_pool2(:, 2) > contribs_threshold, 3)];
positions_pool1_pool2 = [repmat(185, length(indices_pool1_pool2(:, 1)), 1), mean([(indices_pool1_pool2(:, 1)-1)*30+35, (indices_pool1_pool2(:, 2)-1)*12+26], 2)];

% indices_input_pool1 = [m_input_pool1(m_input_pool1(:, 2) > contribs_threshold, 1), m_input_pool1(m_input_pool1(:, 2) > contribs_threshold, 3)];
% positions_input_pool1 = [repmat(65, length(indices_input_pool1(:, 1)), 1), mean([indices_input_pool1(:, 1)+12, (indices_input_pool1(:, 2)-1)*30+15], 2)];

% position = cat(1, positions_ip1_ip2, positions_pool2_ip1, positions_pool1_pool2, positions_input_pool1);
% position = cat(1, positions_ip1_ip2, positions_pool2_ip1, positions_pool1_pool2);

position_max_blobs = cat(1, [485 10], [365 10], [245 10], [125 10]);
position = cat(1, positions_ip1_ip2, positions_pool2_ip1, positions_pool1_pool2, position_max_blobs);

% write value to the corresponding links
M_value = insertText(M, position, value, 'FontSize', 10, 'BoxOpacity', 0.0, 'AnchorPoint', 'Center', 'Textcolor', 'white');

% figure; 
imshow(M_value); 
% title(['label ', num2str(l-1)]);

% pause(2);
% l = l + 1;
end