function [xc, yc] = simple_line(x1, y1, x2, y2)

% distances according to both axes
xn = abs(x2 - x1);
yn = abs(y2 - y1);

% interpolate against axis with greater distance between points;
% this guarantees statement in the under the first point!

% vq = interp1(x, v, xq, method) specifies an alternative interpolation method: 
% 'nearest', 'next', 'previous', 'linear', 'spline', 'pchip', or 'cubic'. 
% The default method is 'linear'.
if (xn > yn)
    xc = x1 : sign(x2 - x1) : x2;
    yc = round(interp1([x1 x2], [y1 y2], xc, 'linear'));
else
    yc = y1 : sign(y2 - y1) : y2;
    xc = round(interp1([y1 y2], [x1 x2], yc, 'linear'));
end

end

% 2-D indexes of line are saved in (xc, yc), and
% 1-D indexes are calculated here:
% linkMatrix = zeros([800, 450]);
% ind = sub2ind(size(linkMatrix), xc, yc);
% 
% % % draw line on the image
% linkMatrix(ind) = weight;

% function linkMatrix = simple_line(x1, y1, x2, y2, weight)
% 
% if abs(x1-x2) > abs(y1-y2)
%     if x1 < x2
%         X = x1: x2;
%         x = [x1 x2];
%         y = [y1 y2];
%         Y = interp1(x, y, X);
%         Y = round(Y);
%     else
%         X = x2: x1;
%         x = [x2 x1];
%         y = [y2 y1];
%         Y = interp1(x, y, X);
%         Y = round(Y);
%     end
% else
%     if y1 < y2
%         Y = y1: y2;
%         x = [x1 x2];
%         y = [y1 y2];
%         X = interp1(y, x, Y);
%         X = round(X);
%     else
%         Y = y2: y1;
%         x = [x2 x1];
%         y = [y2 y1];
%         X = interp1(y, x, Y);
%         X = round(X);
%     end
% end
% 
% linkMatrix = zeros([800, 450]);
% for i = 1: length(X)
%     linkMatrix(X(i), Y(i)) = weight;
% end

% function [X, Y] = simple_line(x1, y1, x2, y2)
% 
% if abs(x1-x2) > abs(y1-y2)
%     if x1 < x2
%         X = x1: x2;
%         x = [x1 x2];
%         y = [y1 y2];
%         Y = interp1(x, y, X);
%         Y = round(Y);
%     else
%         X = x2: x1;
%         x = [x2 x1];
%         y = [y2 y1];
%         Y = interp1(x, y, X);
%         Y = round(Y);
%     end
% else
%     if y1 < y2
%         Y = y1: y2;
%         x = [x1 x2];
%         y = [y1 y2];
%         X = interp1(y, x, Y);
%         X = round(X);
%     else
%         Y = y2: y1;
%         x = [x2 x1];
%         y = [y2 y1];
%         X = interp1(y, x, Y);
%         X = round(X);
%     end
% end

%linkMatrix = zeros([800, 450]);
%ind = sub2ind(size(linkMatrix), X, Y);
%linkMatrix(ind) = weight;
%linkMatrix(ind) = linkMatrix(ind) + weight;


% function [xc, yc] = simple_line(x1, y1, x2, y2)
% 
% % distances according to both axes
% xn = abs(x2 - x1);
% yn = abs(y2 - y1);
% 
% % interpolate against axis with greater distance between points;
% % this guarantees statement in the under the first point!
% if (xn > yn)
%     xc = x1 : sign(x2 - x1) : x2;
%     yc = round(interp1([x1 x2], [y1 y2], xc, 'linear'));
%     
% % vq = interp1(x, v, xq, method) specifies an alternative interpolation method: 
% % 'nearest', 'next', 'previous', 'linear', 'spline', 'pchip', or 'cubic'. 
% % The default method is 'linear'.
% 
% else
%     yc = y1 : sign(y2 - y1) : y2;
%     xc = round(interp1([y1 y2], [x1 x2], yc, 'linear'));
% end

% 2-D indexes of line are saved in (xc, yc), and
% 1-D indexes are calculated here:
% linkMatrix = zeros([800, 450]);
% ind = sub2ind(size(linkMatrix), xc, yc);
% 
% % draw line on the image
% linkMatrix(ind) = weight;

%linkMatrix = zeros([800, 450]);