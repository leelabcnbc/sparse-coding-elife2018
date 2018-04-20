% %%This Program correct the scanline shift and XY shift. 
%% interlacing corrector

NumFrame = 186400;

NBlk = 100; %每256个文件分一组
NTS = ceil(NumFrame/NBlk/20);
% %clear all;
dLine=zeros(NTS,NBlk);           %ScanLine shift  
SumAllFrame=zeros(512,512,NBlk);
% 
if matlabpool('size') > 0
    disp(['matlabpool size = ' int2str(matlabpool('size'))]);
else
    matlabpool 4
end
   
relx = 0;
rely = 0;

tic;

for si = 21:21
    si
    parfor fi = 1:NBlk
        Sum20Frame=zeros(512,512);
        fileTP='G:\Tang Two Photon (2013)\数据分析(来自双光子屋数据处理电脑)\F盘\Tang Two Photon\M081661A21G5RV12012-12-11\2013-04-12\Pic\TP image\Pic\TSeries-04122013-1420-675_Cycle00001_CurrentSettings_Ch2_';

        for j=1:20  
            i=(si-1)*NBlk*20+(fi-1)*20+j;
            imN=num2str(i+1000000);
            imfileName=[fileTP imN(2:7) '.tif'];

            imX=imread(imfileName);
            imY=Sum20Frame;
            Sum20Frame(:,:)=double(imX)+double(imY(:,:));
        end 
        
        imY1=Sum20Frame(1:2:511,:);        
        imY2=Sum20Frame(2:2:512,:);

        pA=zeros(256,512,2);
        pA(:,:,1)=imY1(:,:);
        pA(:,:,2)=imY2(:,:);
 
        WIDTH = size(pA,1);
        HEIGHT = size(pA,2);
        LENGTH = 1;
        NFRAME = 2;
        margeSize=20;              
        pA(WIDTH+1:WIDTH+margeSize*2,HEIGHT+1:HEIGHT+margeSize*2,:) = 0;
        pA(margeSize+1:WIDTH+margeSize,margeSize+1:HEIGHT+margeSize,:)=pA(1:WIDTH,1:HEIGHT,:); 
        
        pbase = pA(:,:,1);
        pinput = pA(:,:,2);
        ssz = round(min(size(pbase))/2);
        x = round(size(pA,2)/4);
        y = round(size(pA,1)/4);
        s1 = pinput(y:y+ssz-1,x:x+ssz-1);
        % imshow(s1);
        c = normxcorr2(s1,pbase(ssz-ssz/2+1-30:ssz+ssz/2+30,ssz-ssz/2+1-30:ssz+ssz/2+30));
                
        [max_c, imax] = max(abs(c(:)));
        [ypeak, xpeak] = ind2sub(size(c),imax(1));
       
        xbias = (xpeak-size(s1,2))+1+ssz/2-30;
        ybias = (ypeak-size(s1,1))+1+ssz/2-30;
        
        relx = xbias-x;
        rely = ybias-y;
        if abs(relx)>margeSize || abs(rely)>margeSize
            disp('exception: exceed margeSize!');
            relx = 0;
            rely = 0;
        end
        pinput(margeSize+1:WIDTH+margeSize, margeSize+1:HEIGHT+margeSize) = pinput((margeSize+1:WIDTH+margeSize)-rely,(margeSize+1:HEIGHT+margeSize)-relx);            

        dLine(si,fi)=relx;     %Calculate the scanline error for each sum of 20 frames.
        
        imY3=zeros(512,512);
        imY3(1:2:511,1:512)=pbase(21:276,21:532);    %Scanline error is corrected for each sum 20 frames, and send to imY3. 
        imY3(2:2:512,1:512)=pinput(21:276,21:532);
        
        imY(1:512,1:512)=SumAllFrame(:,:,fi);
        SumAllFrame(:,:,fi)=double(imY)+double(imY3);             %Get a Sum of all frames after scanline error correlation.
                
    end  
    dLine(si,:)
end

toc;

%This SumAllFrame will be used as a reference for XY correlation.
imY1=sum(SumAllFrame,3)/NBlk/20;%/NTS;  

% % 
figure(2);
imshow(imY1/5000); 

% load imY1 imY1;
% 
% NTS = ceil(NumFrame/NBlk);
% SumAllFrameXYCorr = zeros(512,512,NBlk);
% 
% imYB1 = zeros(512,512,NBlk);
% for fi=1:NBlk
%     imYB1(:,:,fi) = imY1;
% end
% 
% dLine1=zeros(NTS,NBlk);
% % for i=1:NTS
% %     for j=1:NBlk
% %         dLine1(i,j)=dLine(ceil(i/20/NBlk), ceil((i/20/NBlk-ceil(i/20/NBlk)+1)*NBlk));
% %     end
% % end
% % 
% % save dLine1 dLine1;
% 
% tic;
% for si=1:NTS
%     si
%     parfor fi=1:NBlk
%         
%         fileTP='F:\Tang Data Analysis\20130412\Pic\TSeries-04122013-1420-675_Cycle00001_CurrentSettings_Ch2_';
%         i = (si-1)*NBlk+fi;
% 
%         imN=num2str(i+1000000);
%         imfileName=[fileTP imN(2:7) '.tif'];
% 
%         imX=imread(imfileName);
%         %imX=rand(512,512);
%         imY2=double(imX);
%         
%         pA=zeros(512,512,2);
%         pA(:,:,1)=imYB1(:,:,fi);
%         pA(:,:,2)=imY2;
% 
%         % shift correction
%  
%         WIDTH = size(pA,1);
%         HEIGHT = size(pA,2);
%         LENGTH = 1;
%         NFRAME = 2;
%         margeSize=20;
%         pA(WIDTH+1:WIDTH+margeSize*2,HEIGHT+1:HEIGHT+margeSize*2,:) = 0;
%         pA(margeSize+1:WIDTH+margeSize,margeSize+1:HEIGHT+margeSize,:)=pA(1:WIDTH,1:HEIGHT,:);
%         %disp('pA marge added');  
%     
%         %%
%         pbase = pA(:,:,1);
%         pinput = pA(:,:,2);
%         ssz = round(min(size(pbase))/2);
%                 x = round(size(pA,2)/4);
%                 y = round(size(pA,1)/4);
%                 s1 = pinput(y:y+ssz-1,x:x+ssz-1);
%                 % imshow(s1);
%                 %c = normxcorr2(s1,pbase);
%                 c = normxcorr2(s1,pbase(ssz-ssz/2+1-30:ssz+ssz/2+30,ssz-ssz/2+1-30:ssz+ssz/2+30));
%                 
%                 [max_c, imax] = max(abs(c(:)));
%                 [ypeak, xpeak] = ind2sub(size(c),imax(1));
%         
%                 xbias = (xpeak-size(s1,2))+1+ssz/2-30;
%                 ybias = (ypeak-size(s1,1))+1+ssz/2-30;
% 
%                 % s0 = pbase(ybias:(ybias+ssz-1),xbias:(xbias+ssz-1));
%                 % figure,imshow(s0);
%                 relx = xbias-x;
%                 rely = ybias-y;
%                 if abs(relx)>margeSize || abs(rely)>margeSize
%                     disp('exception: exceed margeSize!');
%                     relx = 0;
%                     rely = 0;
%                 end
%                 pinput(margeSize+1:WIDTH+margeSize, margeSize+1:HEIGHT+margeSize) = pinput((margeSize+1:WIDTH+margeSize)-rely,(margeSize+1:HEIGHT+margeSize)-relx);
%                 imY3=zeros(512,512);
%         imY3(1:2:511,1:512)=pinput(21:2:531,21:532);
%         imY3(2:2:512,1:512)=pinput(22:2:532,21-dLine1(si, fi):532-dLine1(si, fi));      %dLine is used for scanline correlation.
%         imwrite(imY3/10000,['F:\Tang Data Analysis\20130412\PicCorrImage\',num2str((si-1)*NBlk+fi)],'tif')       %Save corrected frames
% 
%         imY=SumAllFrameXYCorr(:,:,fi);
%         SumAllFrameXYCorr(:,:,fi) = double(imY)+double(imY3);
%     end
% end
% 
% toc;
% 
% figure(3);
% imshow(sum(SumAllFrameXYCorr,3)/NumFrame/5000);


       


