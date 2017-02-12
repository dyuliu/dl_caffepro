function [h_train, h_test] = plot_curve(st_training_data, s_color)
%PLOT_CURVE Plot a training curve
%   st_training_data: a struct with fields {x_train, y_train, x_test,
%   y_test}
%   c_color: a string to specify line color
    
    h_train = []; h_test = [];

    filter_window_train = 21; filter_window_test = 9;
    if isfield(st_training_data, 'x_train')
        hold on 
        if ischar(s_color)
            h_train = plot(st_training_data.x_train, medfilt1(st_training_data.y_train, filter_window_train), s_color);
        else 
            h_train = plot(st_training_data.x_train, medfilt1(st_training_data.y_train, filter_window_train), 'Color', s_color);
        end
    end
    
    if isfield(st_training_data, 'x_test')
        hold on
        if ischar(s_color)
            h_test = plot(st_training_data.x_test, medfilt1(st_training_data.y_test, filter_window_test), s_color, 'LineWidth', 2);
        else
            h_test = plot(st_training_data.x_test, medfilt1(st_training_data.y_test, filter_window_test), 'Color', s_color, 'LineWidth', 2);
        end
    end
end

