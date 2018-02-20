function image_visual_ORION(mn10_data, ORION_version, label_index, K, count_flag, kernel_level, node_level, sum_norm_flag, max_norm_flag, renorm_flag)

% label_index = label_0_pose_cell{l};

% initialize the network with 660 * 530 pixel (kernel level)
M = zeros([660, 530]);
M(:, 1: 9) = 0.1;
M(:, 131: 139) = 0.1;
M(:, 261: 269) = 0.1;
M(:, 391: 399) = 0.1;
M(:, 521: 530) = 0.1;
M(1: 20, 1: 530) = 0.1;
M((40: 20: 640), 131: 139) = 0.2;
M((40: 20: 640), 261: 269) = 0.2;
M((25: 5: 655), 391: 399) = 0.2;
M((84: 64: 586), 521: 529) = 0.2;

% initialize the layer-link matrices
m_link_CONV1A = zeros([660, 530]);
m_link_CONV2A = zeros([660, 530]);
m_link_FC6 = zeros([660, 530]);
m_link_FC8 = zeros([660, 530]);

% call function contributions to get the matrix_contribs
mat_contribs = contributions_ORION(mn10_data, ORION_version, label_index, K, kernel_level, node_level);

% concatenate the mat_contribs, here the links based on the contribution values
matrix_contribs = [];
for i = 1: K
    matrix_contribs = [matrix_contribs; mat_contribs(:, :, i)];
end

% Visualization based on value or frequency
if count_flag == true
    matrix_contribs(:, 2: 2: end) = 1;
end

% accumulate the contributions in layer conv1a (back-projection to the input image)
[CONV1A, ~, i_conv1a] = unique([matrix_contribs(:, 1), matrix_contribs(:, 3)], 'rows');
m_input_conv1a = zeros([length(CONV1A(:, 1)), 3]);
m_input_conv1a(:, 1) = CONV1A(:, 1);
m_input_conv1a(:, 3) = CONV1A(:, 2);
for i = 1: length(CONV1A(:, 1))
    compoundCondInd = (matrix_contribs(:, 1) == m_input_conv1a(i, 1)) & (matrix_contribs(:, 3) == m_input_conv1a(i, 3));
    m_input_conv1a(i, 2) = sum(matrix_contribs(compoundCondInd, 2));
end

% accumulate the contributions in layer conv2a
[POOL2A, ~, i_pool2a] = unique([matrix_contribs(:, 3), matrix_contribs(:, 5)], 'rows');
m_conv1a_pool2a = zeros([length(POOL2A(:, 1)), 3]);
m_conv1a_pool2a(:, 1) = POOL2A(:, 1);
m_conv1a_pool2a(:, 3) = POOL2A(:, 2);
for i = 1: length(POOL2A(:, 1))
    compoundCondInd = (matrix_contribs(:, 3) == m_conv1a_pool2a(i, 1)) & (matrix_contribs(:, 5) == m_conv1a_pool2a(i, 3));
    m_conv1a_pool2a(i, 2) = sum(matrix_contribs(compoundCondInd, 4));
end

% accumulate the contributions in layer fc6
[FC6, ~, i_fc6] = unique([matrix_contribs(:, 5), matrix_contribs(:, 7)], 'rows');
m_pool2a_fc6 = zeros([length(FC6(:, 1)), 3]);
m_pool2a_fc6(:, 1) = FC6(:, 1);
m_pool2a_fc6(:, 3) = FC6(:, 2);
for i = 1: length(FC6(:, 1))
    compoundCondInd = (matrix_contribs(:, 5) == m_pool2a_fc6(i, 1)) & (matrix_contribs(:, 7) == m_pool2a_fc6(i, 3));
    m_pool2a_fc6(i, 2) = sum(matrix_contribs(compoundCondInd, 6));
end

% accumulate the contributions in layer fc8
[FC8, ~, i_fc8] = unique([matrix_contribs(:, 7), matrix_contribs(:, 9)], 'rows');
m_fc6_fc8 = zeros([length(FC8(:, 1)), 3]);
m_fc6_fc8(:, 1) = FC8(:, 1);
m_fc6_fc8(:, 3) = FC8(:, 2);
for i = 1: length(FC8(:, 1))
    compoundCondInd = (matrix_contribs(:, 7) == m_fc6_fc8(i, 1)) & (matrix_contribs(:, 9) == m_fc6_fc8(i, 3));
    m_fc6_fc8(i, 2) = sum(matrix_contribs(compoundCondInd, 8));
end

% normalize the contributions in layer conv1a, conv2a, fc6, fc8 separately (sum or max)
if sum_norm_flag == true
    m_input_conv1a(:, 2) = m_input_conv1a(:, 2) / sum(m_input_conv1a(:, 2));
    m_conv1a_pool2a(:, 2) = m_conv1a_pool2a(:, 2) / sum(m_conv1a_pool2a(:, 2));
    m_pool2a_fc6(:, 2) = m_pool2a_fc6(:, 2) / sum(m_pool2a_fc6(:, 2));
    m_fc6_fc8(:, 2) = m_fc6_fc8(:, 2) / sum(m_fc6_fc8(:, 2));
elseif max_norm_flag == true
    m_input_conv1a(:, 2) = m_input_conv1a(:, 2) / max(m_input_conv1a(:, 2));
    m_conv1a_pool2a(:, 2) = m_conv1a_pool2a(:, 2) / max(m_conv1a_pool2a(:, 2));
    m_pool2a_fc6(:, 2) = m_pool2a_fc6(:, 2) / max(m_pool2a_fc6(:, 2));
    m_fc6_fc8(:, 2) = m_fc6_fc8(:, 2) / max(m_fc6_fc8(:, 2));
end

% add some switch parameters to control whether to re-normalize or not (contribution propagation)
if renorm_flag == true
    unique_fc6 = unique(m_pool2a_fc6(:, 3));
    unique_pool2a = unique(m_conv1a_pool2a(:, 3));
    unique_conv1a = unique(m_input_conv1a(:, 3));

    for i = 1: length(unique_fc6)
        m_pool2a_fc6(m_pool2a_fc6(:, 3) == unique_fc6(i), 2) = m_pool2a_fc6(m_pool2a_fc6(:, 3) == unique_fc6(i), 2) * sum(m_fc6_fc8(m_fc6_fc8(:, 1) == unique_fc6(i), 2)) / sum(m_pool2a_fc6(m_pool2a_fc6(:, 3) == unique_fc6(i), 2));
    end

    for j = 1: length(unique_pool2a)
        m_conv1a_pool2a(m_conv1a_pool2a(:, 3) == unique_pool2a(j), 2) = m_conv1a_pool2a(m_conv1a_pool2a(:, 3) == unique_pool2a(j), 2) * sum(m_pool2a_fc6(m_pool2a_fc6(:, 1) == unique_pool2a(j), 2)) / sum(m_conv1a_pool2a(m_conv1a_pool2a(:, 3) == unique_pool2a(j), 2));
    end

    for p = 1: length(unique_conv1a)
        m_input_conv1a(m_input_conv1a(:, 3) == unique_conv1a(p), 2) = m_input_conv1a(m_input_conv1a(:, 3) == unique_conv1a(p), 2) * sum(m_conv1a_pool2a(m_conv1a_pool2a(:, 1) == unique_conv1a(p), 2)) / sum(m_input_conv1a(m_input_conv1a(:, 3) == unique_conv1a(p), 2));
    end
end

% find the most important node in each layer, always keep the first largest node
[unique_fc8, ~, index_fc8] = unique(m_fc6_fc8(:, 3));
accum_fc8 = [unique_fc8, accumarray(index_fc8, m_fc6_fc8(:, 2))];
max_index_fc8 = find(accum_fc8(:, 2) == max(accum_fc8(:, 2)));
max_fc8 = accum_fc8(max_index_fc8(1));

[unique_fc6, ~, index_fc6] = unique(m_pool2a_fc6(:, 3));
accum_fc6 = [unique_fc6, accumarray(index_fc6, m_pool2a_fc6(:, 2))];
max_index_fc6 = find(accum_fc6(:, 2) == max(accum_fc6(:, 2)));
max_fc6 = accum_fc6(max_index_fc6(1));

[unique_pool2a, ~, index_pool2a] = unique(m_conv1a_pool2a(:, 3));
accum_pool2a = [unique_pool2a, accumarray(index_pool2a, m_conv1a_pool2a(:, 2))];
max_index_pool2a = find(accum_pool2a(:, 2) == max(accum_pool2a(:, 2)));
max_pool2a = accum_pool2a(max_index_pool2a(1));    % what if two nodes have the same value?

[unique_conv1a, ~, index_conv1a] = unique(m_input_conv1a(:, 3));
accum_conv1a = [unique_conv1a, accumarray(index_conv1a, m_input_conv1a(:, 2))];
max_index_conv1a = find(accum_conv1a(:, 2) == max(accum_conv1a(:, 2)));
max_conv1a = accum_conv1a(max_index_conv1a(1));

% get the link for layer fc8
for u = 1: length(m_fc6_fc8(:, 1))   
    [rows_link_fc8, cols_link_fc8] = bresenham((m_fc6_fc8(u, 1)-1)*5+22.5, 401, (m_fc6_fc8(u, 3)-1)*64+52, 520);
    for m = 1: length(rows_link_fc8)
        m_link_FC8(rows_link_fc8(m), cols_link_fc8(m)) = m_link_FC8(rows_link_fc8(m), cols_link_fc8(m)) + m_fc6_fc8(u, 2);
    end
end

% get the link for layer fc6
for v = 1: length(m_pool2a_fc6(:, 1))
    [rows_link_fc6, cols_link_fc6] = bresenham((m_pool2a_fc6(v, 1)-1)*20+30, 271, (m_pool2a_fc6(v, 3)-1)*5+22.5, 390);
    for m = 1: length(rows_link_fc6)
        m_link_FC6(rows_link_fc6(m), cols_link_fc6(m)) = m_link_FC6(rows_link_fc6(m), cols_link_fc6(m)) + m_pool2a_fc6(v, 2);
    end
end

% get the link for layer conv2a
for w = 1: length(m_conv1a_pool2a(:, 1))
    [rows_conv2a, cols_conv2a] = bresenham((m_conv1a_pool2a(w, 1)-1)*20+30, 141, (m_conv1a_pool2a(w, 3)-1)*20+30, 260);
    for m = 1: length(rows_conv2a)
        m_link_CONV2A(rows_conv2a(m), cols_conv2a(m)) = m_link_CONV2A(rows_conv2a(m), cols_conv2a(m)) + m_conv1a_pool2a(w, 2);
    end
end

%     % get the link for layer conv1a (back-projection to the input image)
%     for x = 1: length(m_input_conv1a(:, 1))
%         [rows_conv1a, cols_conv1a] = bresenham(m_input_conv1a(x, 1)+228, 51, (m_input_conv1a(x, 3)-1)*100+50, 650);
%         for m = 1: length(rows_conv1a)
%             m_link_CONV1A(rows_conv1a(m), cols_conv1a(m)) = m_link_CONV1A(rows_conv1a(m), cols_conv1a(m)) + m_input_conv1a(x, 2);
%         end
%     end

M = M + m_link_FC8 + m_link_FC6 + m_link_CONV2A;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Write significant contributions on the figures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1. Collect the significant contributions value (like larger than 0.1)
contribs_threshold = 0.1;
value_fc6_fc8 = m_fc6_fc8(m_fc6_fc8(:, 2) > contribs_threshold, 2);
value_pool2a_fc6 = m_pool2a_fc6(m_pool2a_fc6(:, 2) > contribs_threshold, 2);
value_conv1a_pool2a = m_conv1a_pool2a(m_conv1a_pool2a(:, 2) > contribs_threshold, 2);
%     value_input_conv1a = m_input_conv1a(m_input_conv1a(:, 2) > contribs_threshold, 2);
%     value = cat(1, value_fc6_fc8, value_pool2a_fc6, value_conv1a_pool2a, value_input_conv1a);

% concatenate the value of important nodes
max_blobs = cat(1, max_fc8, max_fc6, max_pool2a, max_conv1a);
value = cat(1, value_fc6_fc8, value_pool2a_fc6, value_conv1a_pool2a, max_blobs);

% 2. Collect the corresponding positions
indices_fc6_fc8 = [m_fc6_fc8(m_fc6_fc8(:, 2) > contribs_threshold, 1), m_fc6_fc8(m_fc6_fc8(:, 2) > contribs_threshold, 3)];
positions_fc6_fc8 = [repmat(460, length(indices_fc6_fc8(:, 1)), 1), mean([(indices_fc6_fc8(:, 1)-1)*5+22.5, (indices_fc6_fc8(:, 2)-1)*64+52], 2)];

indices_pool2a_fc6 = [m_pool2a_fc6(m_pool2a_fc6(:, 2) > contribs_threshold, 1), m_pool2a_fc6(m_pool2a_fc6(:, 2) > contribs_threshold, 3)];
positions_pool2a_fc6 = [repmat(330, length(indices_pool2a_fc6(:, 1)), 1), mean([(indices_pool2a_fc6(:, 1)-1)*20+30, (indices_pool2a_fc6(:, 2)-1)*5+22.5], 2)];

indices_conv1a_pool2a = [m_conv1a_pool2a(m_conv1a_pool2a(:, 2) > contribs_threshold, 1), m_conv1a_pool2a(m_conv1a_pool2a(:, 2) > contribs_threshold, 3)];
positions_conv1a_pool2a = [repmat(200, length(indices_conv1a_pool2a(:, 1)), 1), mean([(indices_conv1a_pool2a(:, 1)-1)*20+30, (indices_conv1a_pool2a(:, 2)-1)*20+30], 2)];

%     indices_input_conv1a = [m_input_conv1a(m_input_conv1a(:, 2) > contribs_threshold, 1), m_input_conv1a(m_input_conv1a(:, 2) > contribs_threshold, 3)];
%     positions_input_conv1a = [repmat(350, length(indices_input_conv1a(:, 1)), 1), mean([indices_input_conv1a(:, 1)+228, (indices_input_conv1a(:, 2)-1)*100+50], 2)];
%     
%     position = cat(1, positions_fc6_fc8, positions_pool2a_fc6, positions_conv1a_pool2a, positions_input_conv1a);
position_max_blobs = cat(1, [525 10], [395 10], [265 10], [135 10]);
position = cat(1, positions_fc6_fc8, positions_pool2a_fc6, positions_conv1a_pool2a, position_max_blobs);

% write value to the corresponding links
M_value = insertText(M, position, value, 'FontSize', 10, 'BoxOpacity', 0.0, 'AnchorPoint', 'Center', 'Textcolor', 'white');

imshow(M_value); 
end
