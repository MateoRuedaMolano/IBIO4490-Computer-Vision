function plot_eval_ourMethods(evalDir,evalDir_Our1,evalDir_Our2,col1,col2,col3)
% plot evaluation results.
% Pablo Arbelaez <arbelaez@eecs.berkeley.edu>
if nargin<4, col1 = 'r'; end
fwrite(2,sprintf('\n%s\n',evalDir));
if exist(fullfile(evalDir,'eval_bdry_thr.txt'),'file'),
    %     Plotting UCM results:
    open('isoF.fig');
    hold on
    prvals = dlmread(fullfile(evalDir,'eval_bdry_thr.txt')); % thresh,r,p,f
    f=find(prvals(:,2)>=0.01);
    prvals = prvals(f,:);
    evalRes = dlmread(fullfile(evalDir,'eval_bdry.txt'));
    if size(prvals,1)>1,
        plot(prvals(1:end,2),prvals(1:end,3),col1,'LineWidth',3,'DisplayName','UCM');
    else
        plot(evalRes(2),evalRes(3),'o','MarkerFaceColor',col1,'MarkerEdgeColor',col1,'MarkerSize',8,'DisplayName','UCM');
    end
    fprintf('Boundary\n');
    fprintf('ODS: F( %1.2f, %1.2f ) = %1.2f   [th = %1.2f]\n',evalRes(2:4),evalRes(1));
    fprintf('OIS: F( %1.2f, %1.2f ) = %1.2f\n',evalRes(5:7));
    fprintf('Area_PR = %1.2f\n\n',evalRes(8));
    
    %   Plotting out results:
    %     For the first method
    fwrite(2,sprintf('\n%s\n',evalDir_Our1));
    if exist(fullfile(evalDir_Our1,'eval_bdry_thr.txt'),'file'),
        if nargin<5, col2 = 'b'; end
        prvals_our1 = dlmread(fullfile(evalDir_Our1,'eval_bdry_thr.txt')); % thresh,r,p,f
        f_our1=find(prvals_our1(:,2)>=0.01);
        prvals_our1 = prvals_our1(f_our1,:);
        evalRes_our1 = dlmread(fullfile(evalDir_Our1,'eval_bdry.txt'));
        if size(prvals_our1,1)>1,
            plot(prvals_our1(1:end,2),prvals_our1(1:end,3),col2,'LineWidth',3,'DisplayName','Our Method1');
        else
            plot(evalRes_our1(2),evalRes_our1(3),'o','MarkerFaceColor',col2,'MarkerEdgeColor',col2,'MarkerSize',8,'DisplayName','Our Method1');
        end
        fprintf('Boundary-Our method1\n');
        fprintf('ODS: F( %1.2f, %1.2f ) = %1.2f   [th = %1.2f]\n',evalRes_our1(2:4),evalRes_our1(1));
        fprintf('OIS: F( %1.2f, %1.2f ) = %1.2f\n',evalRes_our1(5:7));
        fprintf('Area_PR = %1.2f\n',evalRes_our1(8));
    end
    %     For the Second method
    fwrite(2,sprintf('\n%s\n',evalDir_Our2));
    if exist(fullfile(evalDir_Our2,'eval_bdry_thr.txt'),'file'),
        if nargin<6, col3 = 'y'; end
        prvals_our2 = dlmread(fullfile(evalDir_Our2,'eval_bdry_thr.txt')); % thresh,r,p,f
        f_our2=find(prvals_our2(:,2)>=0.01);
        prvals_our2 = prvals_our2(f_our2,:);
        evalRes_our2 = dlmread(fullfile(evalDir_Our2,'eval_bdry.txt'));
        if size(prvals_our2,1)>1,
            plot(prvals_our2(1:end,2),prvals_our2(1:end,3),col3,'LineWidth',3,'DisplayName','Our Method2');
        else
            plot(evalRes_our2(2),evalRes_our2(3),'o','MarkerFaceColor',col3,'MarkerEdgeColor',col3,'MarkerSize',8,'DisplayName','Our Method2');
        end
        fprintf('Boundary-Our method2\n');
        fprintf('ODS: F( %1.2f, %1.2f ) = %1.2f   [th = %1.2f]\n',evalRes_our2(2:4),evalRes_our2(1));
        fprintf('OIS: F( %1.2f, %1.2f ) = %1.2f\n',evalRes_our2(5:7));
        fprintf('Area_PR = %1.2f\n',evalRes_our2(8));
    end
    hold off
end

fwrite(2,sprintf('\n%s\n',evalDir));
if exist(fullfile(evalDir,'eval_cover.txt'),'file'),
    %     Printing UCM results
    evalRes = dlmread(fullfile(evalDir,'eval_cover.txt'));
    fprintf('Region\n');
    fprintf('GT covering: ODS = %1.2f [th = %1.2f]. OIS = %1.2f. Best = %1.2f\n',evalRes(2),evalRes(1),evalRes(3:4));
    evalRes = dlmread(fullfile(evalDir,'eval_RI_VOI.txt'));
    fprintf('Rand Index: ODS = %1.2f [th = %1.2f]. OIS = %1.2f.\n',evalRes(2),evalRes(1),evalRes(3));
    fprintf('Var. Info.: ODS = %1.2f [th = %1.2f]. OIS = %1.2f.\n',evalRes(5),evalRes(4),evalRes(6));
    
    %     Printing our results:
    %     For method1:
    fwrite(2,sprintf('\n%s\n',evalDir_Our1));
    if exist(fullfile(evalDir_Our1,'eval_bdry_thr.txt'),'file'),
        evalRes_our1 = dlmread(fullfile(evalDir_Our1,'eval_cover.txt'));
        fprintf('Region-Our method1\n');
        fprintf('GT covering: ODS = %1.2f [th = %1.2f]. OIS = %1.2f. Best = %1.2f\n',evalRes_our1(2),evalRes_our1(1),evalRes_our1(3:4));
        evalRes_our1 = dlmread(fullfile(evalDir_Our1,'eval_RI_VOI.txt'));
        fprintf('Rand Index: ODS = %1.2f [th = %1.2f]. OIS = %1.2f.\n',evalRes_our1(2),evalRes_our1(1),evalRes_our1(3));
        fprintf('Var. Info.: ODS = %1.2f [th = %1.2f]. OIS = %1.2f.\n',evalRes_our1(5),evalRes_our1(4),evalRes_our1(6));
    end
    %     For method2:
    fwrite(2,sprintf('\n%s\n',evalDir_Our2));
    if exist(fullfile(evalDir_Our2,'eval_bdry_thr.txt'),'file'),
        evalRes_our2 = dlmread(fullfile(evalDir_Our2,'eval_cover.txt'));
        fprintf('Region-Our method2\n');
        fprintf('GT covering: ODS = %1.2f [th = %1.2f]. OIS = %1.2f. Best = %1.2f\n',evalRes_our2(2),evalRes_our2(1),evalRes_our2(3:4));
        evalRes_our2 = dlmread(fullfile(evalDir_Our2,'eval_RI_VOI.txt'));
        fprintf('Rand Index: ODS = %1.2f [th = %1.2f]. OIS = %1.2f.\n',evalRes_our2(2),evalRes_our2(1),evalRes_our2(3));
        fprintf('Var. Info.: ODS = %1.2f [th = %1.2f]. OIS = %1.2f.\n',evalRes_our2(5),evalRes_our2(4),evalRes_our2(6));
    end
end