function input_region_LeNet(images, label_index, K, kernel_level, node_level, Num)

% call function contributions to get the matrix_contribs
[mat_contribs, input_image, input_region, position] = contributions_LeNet(images, label_index, K, kernel_level, node_level);

% plot the most important region of the input image through back-projection
loop_var = sqrt(Num * 2);
figure;
for i = 1: loop_var
    for j = 1: loop_var
        if mod(j, 2) == 0
            subplot(loop_var, loop_var, (i-1)*loop_var+j), imshow(input_region(:, :, (i-1)*2+fix(j/2)));
        else
            subplot(loop_var, loop_var, (i-1)*loop_var+j), imshow(input_image(:, :, i*2-1+fix(j/2))), ...
                rectangle('Position', [position(i*2-1+fix(j/2),2)-0.5 position(i*2-1+fix(j/2),1)-0.5 5 5], 'EdgeColor', 'r');
        end
    end
end
%suptitle(['Subplots of ', num2str(i*j/2), ' input images']);