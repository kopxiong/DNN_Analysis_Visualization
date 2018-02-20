function varargout = visualization_gui(varargin)
% VISUALIZATION_GUI MATLAB code for visualization_gui.fig
%      VISUALIZATION_GUI, by itself, creates a new VISUALIZATION_GUI or raises the existing
%      singleton*.
%
%      H = VISUALIZATION_GUI returns the handle to a new VISUALIZATION_GUI or the handle to
%      the existing singleton*.
%
%      VISUALIZATION_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in VISUALIZATION_GUI.M with the given input arguments.
%
%      VISUALIZATION_GUI('Property','Value',...) creates a new VISUALIZATION_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before visualization_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to visualization_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help visualization_gui

% Last Modified by GUIDE v2.5 03-May-2017 12:28:02

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @visualization_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @visualization_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before visualization_gui is made visible.
function visualization_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to visualization_gui (see VARARGIN)

% Choose default command line output for visualization_gui
handles.output = hObject;

% Store some application data 
handles.images = loadMNISTImages('/home/xiongy/DNN_analysis_project/LeNet5/codes_lenet5/MNIST/train-images-idx3-ubyte');
labels = loadMNISTLabels('/home/xiongy/DNN_analysis_project/LeNet5/codes_lenet5/MNIST/train-labels-idx1-ubyte');

% % pick a small number for debug only
% ind = floor(linspace(1, numel(labels), 1000));
% handles.images = handles.images(:, ind);
% labels = labels(ind);

% find the separated indices for different labels
handles.label_0 = find(labels == 0);
handles.label_1 = find(labels == 1);
handles.label_2 = find(labels == 2);
handles.label_3 = find(labels == 3);
handles.label_4 = find(labels == 4);
handles.label_5 = find(labels == 5);
handles.label_6 = find(labels == 6);
handles.label_7 = find(labels == 7);
handles.label_8 = find(labels == 8);
handles.label_9 = find(labels == 9);

% define some default parameters
handles.count_flag = false;
handles.kernel_level = false;
handles.node_level = false;
handles.sum_norm_flag = false;
handles.max_norm_flag = false;
handles.renorm_flag = false;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes visualization_gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = visualization_gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in norm_popupmenu.
function norm_popupmenu_Callback(hObject, eventdata, handles)
% hObject    handle to norm_popupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns norm_popupmenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from norm_popupmenu

% Determine the selected data set.
val = get(hObject, 'Value');
contents = cellstr(get(hObject,'String'));

% Set current data to the selected data set.
switch contents{val}
    case 'sum' 
       handles.sum_norm_flag = true;
       handles.max_norm_flag = false;
    case 'max' 
       handles.max_norm_flag = true;
       handles.sum_norm_flag = false;
end

% Save the handles structure.
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function norm_popupmenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to norm_popupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','whidata = guidata()te');
end


% --- Executes on selection change in renorm_popupmenu.K
function renorm_popupmenu_Callback(hObject, eventdata, handles)
% hObject    handle to renorm_popupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns renorm_popupmenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from renorm_popupmenu

% Determine the selected data set.
val = get(hObject, 'Value');
contents = cellstr(get(hObject, 'String'));

% Set current data to the selected data set.
switch contents{val}
    case 'renorm' 
       handles.renorm_flag = true;
    case 'unrenorm' 
       handles.renorm_flag = false;
end

% Save the handles structure.
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function renorm_popupmenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to renorm_popupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function top_k_edit_Callback(hObject, eventdata, handles)
% hObject    handle to top_k_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of top_k_edit as text
%        str2double(get(hObject,'String')) returns contents of top_k_edit as a double

handles.K = str2double(get(hObject, 'String'));

guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function top_k_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to top_k_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in label_popupmenu.
function label_popupmenu_Callback(hObject, eventdata, handles)
% hObject    handle to label_popupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns label_popupmenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from label_popupmenu

val = get(hObject, 'Value');
contents = cellstr(get(hObject, 'String'));

% str = get(hObject, 'String');
% handles.label_popupmenu = contents{val};

switch contents{val}
    case 'label_0' 
       handles.label_popupmenu = handles.label_0;
    case 'label_1'
       handles.label_popupmenu = handles.label_1;
    case 'label_2' 
       handles.label_popupmenu = handles.label_2;
    case 'label_3'
       handles.label_popupmenu = handles.label_3;
    case 'label_4' 
       handles.label_popupmenu = handles.label_4;
    case 'label_5'
       handles.label_popupmenu = handles.label_5;
    case 'label_6' 
       handles.label_popupmenu = handles.label_6;
    case 'label_7'
       handles.label_popupmenu = handles.label_7;
    case 'label_8' 
       handles.label_popupmenu = handles.label_8;
    case 'label_9'
       handles.label_popupmenu = handles.label_9;
end

guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function label_popupmenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to label_popupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in visualize_pushbutton.
function visualize_pushbutton_Callback(hObject, eventdata, handles)
% hObject    handle to visualize_pushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

image_visual_LeNet(handles.images, handles.label_popupmenu, handles.K, handles.count_flag, handles.kernel_level, ...
    handles.node_level, handles.sum_norm_flag, handles.max_norm_flag, handles.renorm_flag);

% Update handles structure
guidata(hObject, handles);


function Num_edit_Callback(hObject, eventdata, handles)
% hObject    handle to Num_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Num_edit as text
%        str2double(get(hObject,'String')) returns contents of Num_edit as a double

handles.Num = str2double(get(hObject, 'String'));

guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function Num_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Num_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in input_region_pushbutton.
function input_region_pushbutton_Callback(hObject, eventdata, handles)
% hObject    handle to input_region_pushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% handles.K
% handles.label_popupmenu    % 10*1 cell

% plot(sin(0:.1:10));   % simple example for plotting

input_region_LeNet(handles.images, handles.label_popupmenu, handles.K, handles.kernel_level, handles.node_level, handles.Num);

% % call function contributions to get the matrix_contribs
% [mat_contribs, input_image, input_region, position] = contributions_LeNet(handles.label_popupmenu, handles.K);
% 
% panhandle = uipanel('Position', [32, 50, 390, 362]);
% % pax = {4};

% % plot the most important region of the input image through back-projection
% for i = 1: 8
%     for j = 1: 8
%         if mod(j, 2) == 0
%             pax{(i-1)*8+j} = subplot(8, 8, (i-1)*8+j, 'Parent', panhandle);
%             imshow(input_region(:, :, (i-1)*2+fix(j/2)));
%         else
%             pax{(i-1)*8+j} = subplot(8, 8, (i-1)*8+j, 'Parent', panhandle); 
%             imshow(input_image(:, :, i*2-1+fix(j/2))), rectangle('Position', ...
%                 [position(i*2-1+fix(j/2),2)-0.5 position(i*2-1+fix(j/2),1)-0.5 5 5], 'EdgeColor', 'r');
%         end
%     end
% end
% 
% handles.panhandle = panhandle;
% for i = 1: 8
%     for j = 1: 8
%         handles.pax{(i-1)*8+j} = pax{(i-1)*8+j};
%     end
% end
% suptitle(['50 inputs for label ', num2str(0)]);

% Update handles structure
guidata(hObject, handles);


% --- Executes on selection change in visual_level_popupmenu.
function visual_level_popupmenu_Callback(hObject, eventdata, handles)
% hObject    handle to visual_level_popupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns visual_level_popupmenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from visual_level_popupmenu

% Determine the selected data set.
val = get(hObject, 'Value');
contents = cellstr(get(hObject,'String'));

% Set current data to the selected data set.
switch contents{val}
    case 'kernel' 
       handles.kernel_level = true;
       handles.node_level = false;
    case 'node' 
       handles.node_level = true;
       handles.kernel_level = false;
end

% Save the handles structure.
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function visual_level_popupmenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to visual_level_popupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in count_flag_popupmenu.
function count_flag_popupmenu_Callback(hObject, eventdata, handles)
% hObject    handle to count_flag_popupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns count_flag_popupmenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from count_flag_popupmenu

% Determine the selected data set.
val = get(hObject, 'Value');
contents = cellstr(get(hObject, 'String'));

% Set current data to the selected data set.
switch contents{val}
    case 'value' 
       handles.count_flag = false;
    case 'frequency' 
       handles.count_flag = true;
end

% Save the handles structure.
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function count_flag_popupmenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to count_flag_popupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
