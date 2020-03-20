%% Start with a clean slate
clear all; close all; clc

%% Trying to remove noise from recording project
%% load all data
directories = dir('E:\FakeDesktop\voices\VOiCES_devkit\distant-16k\speech\train\rm1\none');
allVoicesNoNoiseFFT = zeros(8000,7908);
index = 1;
for i=1:length(directories)
    if directories(i).name == "." | directories(i).name == ".."
        continue
    end
    directories(i).name
    voices = dir(strcat(directories(i).folder, '\', directories(i).name));
    for j=1:length(voices)
        if voices(j).name == "." | voices(j).name == ".."
            continue
        end
        [voice,Fs] = audioread(strcat(voices(j).folder, '\', voices(j).name));
        for k=1:floor(length(voice)/Fs)
            if k > 10
                break
            end
            allVoicesNoNoiseFFT(:,index) = fft(voice(1+Fs*(k-1):Fs*k-Fs/2));
            index = index + 1;
        end
    end
end

%% compute SVDs

mean_allVoicesFFT = mean(allVoicesNoNoiseFFT);
allVoicesNoNoiseFFT = allVoicesNoNoiseFFT - mean_allVoicesFFT;
[uf,sf,vf] = svd(allVoicesNoNoiseFFT,'econ');

%% variances
sigf=diag(sf);
figure(2)
subplot(1,2,1), plot(sigf,'ko','Linewidth',[1.5])
subplot(1,2,2), semilogy(sigf,'ko','Linewidth',[1.5])
%% what do some of the modes sound like for the data?
modes = 7000;
approx1 = ifft(uf(:,1:modes)*sf(1:modes,1:modes)*vf(:,1:modes)' + mean_allVoicesFFT);
p = audioplayer(1*approx1(:,10),Fs);
p.play();

%p = audioplayer(ifft(allVoicesNoNoiseFFT(:,1)),Fs);
%p.play();

%% load noisy data
%noisyPath = 'E:\FakeDesktop\voices\VOiCES_devkit\distant-16k\speech\train\rm1\musi\sp0093\Lab41-SRI-VOiCES-rm1-musi-sp0093-ch123172-sg0024-mc01-stu-clo-dg020.wav';
noisyPath = 'E:\FakeDesktop\voices\VOiCES_devkit\distant-16k\speech\train\rm1\musi\sp0083\Lab41-SRI-VOiCES-rm1-musi-sp0083-ch009960-sg0031-mc01-stu-clo-dg110.wav';
%noisyPath = 'E:\FakeDesktop\voices\VOiCES_devkit\distant-16k\speech\train\rm1\musi\sp8713\Lab41-SRI-VOiCES-rm1-musi-sp8713-ch296159-sg0045-mc05-stu-far-dg110.wav';
[noisyVoice,noisyFs] = audioread(noisyPath);

noisyVoiceFFT = zeros(8000,floor(length(noisyVoice)/noisyFs));
for k=1:floor(length(noisyVoice)/noisyFs)
    noisyVoiceFFT(:,2*k-1) = fft(noisyVoice(1+noisyFs*(k-1):noisyFs*k-Fs/2));
    noisyVoiceFFT(:,2*k) = fft(noisyVoice(1+noisyFs*k-Fs/2:noisyFs*k));
end

mean_noisyFFT = mean(noisyVoiceFFT);
noisyVoiceFFT = noisyVoiceFFT - mean_noisyFFT;

%% can we extract noiseless data from FFT?
startmode = 1;
endmode = 4000;
Transform = uf(:,startmode:endmode)';
inverseTransform = uf(:,startmode:endmode);
transformedNoise = (Transform * noisyVoiceFFT)';
revertTransformNoise = inverseTransform*transformedNoise';

%p = audioplayer(noisyVoice, Fs);
%p.play();

%p = audioplayer(ifft(revertTransformNoise),Fs);
%p.play();

original = [];
for i=1:size(revertTransformNoise,2)
    original = [original ifft(revertTransformNoise(:,i))];
end
noiseReducedTrack = reshape(original,size(original,1)*size(original,2),1);
p = audioplayer(noiseReducedTrack,Fs);
p.play();

%%
cleanVersionPath = 'E:\FakeDesktop\voices\VOiCES_devkit\distant-16k\speech\train\rm1\none\sp0083\Lab41-SRI-VOiCES-rm1-none-sp0083-ch009960-sg0031-mc01-stu-clo-dg110.wav';
%cleanVersionPath = 'E:\FakeDesktop\voices\VOiCES_devkit\distant-16k\speech\train\rm1\sp8713\Lab41-SRI-VOiCES-rm1-none-sp8713-ch296159-sg0045-mc05-stu-far-dg110.wav';
[cleanVoice,cleanFs] = audioread(cleanVersionPath);
figure(2)
subplot(3,1,1)
plot(abs(cleanVoice));
set(gca,'Xlim',[0 250000])
title('No Noise Signal');
subplot(3,1,2)
plot(abs(noiseReducedTrack));
set(gca,'Xlim',[0 250000])
title('Attempted Noise Reduction');
subplot(3,1,3)
plot(abs(noisyVoice));
set(gca,'Xlim',[0 250000])
title('Noisy Signal');

p = audioplayer(noisyVoice,Fs);
p.play();

%%
figure(3)
subplot(3,1,1)
plot(abs(fftshift(fft(cleanVoice(1:Fs)))));
title('No Noise Signal');
subplot(3,1,2)
plot(abs(fftshift(fft(noiseReducedTrack(1:Fs)))));
title('Attempted Noise Reduction');
subplot(3,1,3)
plot(abs(fftshift(fft(noisyVoice(1:Fs)))));
title('Noisy Signal');

%% Categorize Human speech vs. noise

babbleFolder = dir('E:\FakeDesktop\voices\VOiCES_devkit\distant-16k\distractors\rm3\babb');
%babbleSounds = [];
babbleSounds = zeros(57351,380);

index = 1;
% skip first 3 elements: '.', '..', and the cross validation data
for i=4:length(babbleFolder)
    if babbleFolder(i).name == "." | babbleFolder(i).name == ".."
        continue
    end
    
    % each file is 30+ minutes long. Take some 5 second chunks from each
    [m,Fs] = audioread(strcat(babbleFolder(i).folder, '\', babbleFolder(i).name));
    for j=1:20
        babbleSounds(:,index) = reshape(spectrogram(m(j*Fs*30:2:j*Fs*30+Fs*5+1,1)),8193*7,1);
        index = index + 1;
    end
end

musicFolder = dir('E:\FakeDesktop\voices\VOiCES_devkit\distant-16k\distractors\rm3\musi');
musicSounds = zeros(57351,380);

index = 1;
% skip first 3 elements: '.', '..', and the cross validation data
for i=4:length(musicFolder)
    if musicFolder(i).name == "." | musicFolder(i).name == ".."
        continue
    end
    
    % each file is 30+ minutes long. Take some 5 second chunks from each
    [m,Fs] = audioread(strcat(musicFolder(i).folder, '\', musicFolder(i).name));
    for j=1:20
        musicSounds(:,index) = reshape(spectrogram(m(j*Fs*30:2:j*Fs*30+Fs*5+1,1)),8193*7,1);
        index = index + 1;
    end
end

tvFolder = dir('E:\FakeDesktop\voices\VOiCES_devkit\distant-16k\distractors\rm3\tele');
tvSounds = zeros(57351,380);

index = 1;
% skip first 3 elements: '.', '..', and the cross validation data
for i=4:length(tvFolder)
    if tvFolder(i).name == "." | tvFolder(i).name == ".."
        continue
    end
    
    % each file is 30+ minutes long. Take some 5 second chunks from each
    [m,Fs] = audioread(strcat(tvFolder(i).folder, '\', tvFolder(i).name));
    for j=1:20
        tvSounds(:,index) = reshape(spectrogram(m(j*Fs*30:2:j*Fs*30+Fs*5+1,1)),8193*7,1);
        index = index + 1;
    end
end

directories = dir('E:\FakeDesktop\voices\VOiCES_devkit\distant-16k\speech\train\rm1\none');
allVoicesNoNoiseSPCTR = zeros(57351,380);
index = 1;
for i=1:length(directories)
    if directories(i).name == "." | directories(i).name == ".."
        continue
    end
    %directories(i).name
    voices = dir(strcat(directories(i).folder, '\', directories(i).name));
    for j=1:2:length(voices)
        if voices(j).name == "." | voices(j).name == ".."
            continue
        end
        [m,Fs] = audioread(strcat(voices(j).folder, '\', voices(j).name));

        allVoicesNoNoiseSPCTR(:,index) = reshape(spectrogram(m(Fs*1:2:Fs*6+1,1)),8193*7,1);
        index = index + 1;
        if index == 381
            break;
        end
    end
    if index == 381
        break;
    end
end

allSounds = abs([babbleSounds musicSounds tvSounds allVoicesNoNoiseSPCTR]);
mean_allSounds = mean(allSounds);
allSoundsCentered = allSounds - mean_allSounds;
[u,s,v] = svd(allSoundsCentered,'econ');

sig=diag(s);
figure(6)
subplot(1,2,1), plot(sig,'ko','Linewidth',[1.5])
subplot(1,2,2), semilogy(sig,'ko','Linewidth',[1.5])
%%
endmodelist = [1 2 3 4 5 7 13 17 25:5:1515];
xexpected=string(zeros(240,1));
xexpected(1:60)="babble";
xexpected(61:120)="music";
xexpected(121:180)="tv";
xexpected(181:end)="voice";
ctrain=string(zeros(320*4,1));
ctrain(1:320)="babble";
ctrain(321:640)="music";
ctrain(641:960)="tv";
ctrain(961:end)="voice";
trainAccuracies = zeros(1,length(endmodelist));
    
for i=1:length(endmodelist)
    startmode = 1;
    endmodes = endmodelist(i);

    random_babble = randperm(380);
    random_music = randperm(380);
    random_tv = randperm(380);
    random_voice = randperm(380);

    babble = v(1:380,startmode:endmodes);
    music = v(381:760,startmode:endmodes);
    tv = v(761:1140,startmode:endmodes);
    voice = v(1141:1520,startmode:endmodes);

    xtrain=[babble(random_babble(1:320),:); music(random_music(1:320),:); tv(random_tv(1:320),:); voice(random_voice(1:320),:)];
    xtest=[babble(random_babble(321:end),:); music(random_music(321:end),:); tv(random_tv(321:end),:); voice(random_voice(321:end),:)];

    nb=fitcnb(xtrain,ctrain);
    trainAccuracies(i) = sum(nb.predict(xtest)==xexpected)/length(xexpected);
end

figure(7)
plot(endmodelist, trainAccuracies);

%%
endmodelist = [1 2 3 4 5 7 13 17 25:5:400];
xexpected=string(zeros(240,1));
xexpected(1:60)="babble";
xexpected(61:120)="music";
xexpected(121:180)="tv";
xexpected(181:end)="voice";
ctrain=string(zeros(320*4,1));
ctrain(1:320)="babble";
ctrain(321:640)="music";
ctrain(641:960)="tv";
ctrain(961:end)="voice";
trainAccuracies = zeros(1,length(endmodelist));
    
for j=1:5
for i=1:length(endmodelist)
    startmode = 1;
    endmodes = endmodelist(i);

    random_babble = randperm(380);
    random_music = randperm(380);
    random_tv = randperm(380);
    random_voice = randperm(380);

    babble = v(1:380,startmode:endmodes);
    music = v(381:760,startmode:endmodes);
    tv = v(761:1140,startmode:endmodes);
    voice = v(1141:1520,startmode:endmodes);

    xtrain=[babble(random_babble(1:320),:); music(random_music(1:320),:); tv(random_tv(1:320),:); voice(random_voice(1:320),:)];
    xtest=[babble(random_babble(321:end),:); music(random_music(321:end),:); tv(random_tv(321:end),:); voice(random_voice(321:end),:)];

    nb=fitcnb(xtrain,ctrain);
    trainAccuracies(i) = trainAccuracies(i) + sum(nb.predict(xtest)==xexpected)/length(xexpected);
end
end

figure(8)
plot(endmodelist, trainAccuracies/5);

%% startmode
startmodelist = [1 2 3 4 5 7 13 17 25:5:85];
xexpected=string(zeros(240,1));
xexpected(1:60)="babble";
xexpected(61:120)="music";
xexpected(121:180)="tv";
xexpected(181:end)="voice";
ctrain=string(zeros(320*4,1));
ctrain(1:320)="babble";
ctrain(321:640)="music";
ctrain(641:960)="tv";
ctrain(961:end)="voice";
trainAccuracies = zeros(1,length(startmodelist));
    
for j=1:5
for i=1:length(startmodelist)
    startmode = startmodelist(i);
    endmodes = 85;

    random_babble = randperm(380);
    random_music = randperm(380);
    random_tv = randperm(380);
    random_voice = randperm(380);

    babble = v(1:380,startmode:endmodes);
    music = v(381:760,startmode:endmodes);
    tv = v(761:1140,startmode:endmodes);
    voice = v(1141:1520,startmode:endmodes);

    xtrain=[babble(random_babble(1:320),:); music(random_music(1:320),:); tv(random_tv(1:320),:); voice(random_voice(1:320),:)];
    xtest=[babble(random_babble(321:end),:); music(random_music(321:end),:); tv(random_tv(321:end),:); voice(random_voice(321:end),:)];

    nb=fitcnb(xtrain,ctrain);
    trainAccuracies(i) = trainAccuracies(i) + sum(nb.predict(xtest)==xexpected)/length(xexpected);
end
end

figure(9)
plot(startmodelist, trainAccuracies/5);

%% retrain model with best start and end
startmode = 1;
endmodes = 85;
Transform = s(startmode:endmodes,startmode:endmodes)\u(:,startmode:endmodes)';
random_babble = randperm(380);
random_music = randperm(380);
random_tv = randperm(380);
random_voice = randperm(380);

babble = v(1:380,startmode:endmodes);
music = v(381:760,startmode:endmodes);
tv = v(761:1140,startmode:endmodes);
voice = v(1141:1520,startmode:endmodes);

xtrain=[babble(random_babble(1:320),:); music(random_music(1:320),:); tv(random_tv(1:320),:); voice(random_voice(1:320),:)];

nb=fitcnb(xtrain,ctrain);

%%
correctBayes = 0;
% take a completely unused Music noise sample and test against it
[m,Fs] = audioread('E:\FakeDesktop\voices\VOiCES_devkit\distant-16k\distractors\rm3\musi\Lab41-SRI-VOiCES-rm3-musi-mc01-stu-clo.wav');
testMusic = zeros(8193*7, 20);
for j=1:20
    testMusic(:,j) = reshape(spectrogram(m(j*Fs*30:2:j*Fs*30+Fs*5+1,1)),8193*7,1);
end
testMusic = abs(testMusic);
testMusic = testMusic - mean(testMusic);
transformedMusic = (Transform * testMusic)';
musicPredictions = nb.predict(transformedMusic);
correctBayes = correctBayes + sum(musicPredictions=="music");

% repeat for Babble
[m,Fs] = audioread('E:\FakeDesktop\voices\VOiCES_devkit\distant-16k\distractors\rm3\babb\Lab41-SRI-VOiCES-rm3-babb-mc01-stu-clo.wav');
testBabble = zeros(8193*7, 20);
for j=1:20
    testBabble(:,j) = reshape(spectrogram(m(j*Fs*30:2:j*Fs*30+Fs*5+1,1)),8193*7,1);
end
testBabble = abs(testBabble);
testBabble = testBabble - mean(testBabble);
transformedBabble = (Transform * testBabble)';
babblePredictions = nb.predict(transformedBabble);
correctBayes = correctBayes + sum(babblePredictions=="babble");

% repeat for TV
[m,Fs] = audioread('E:\FakeDesktop\voices\VOiCES_devkit\distant-16k\distractors\rm3\tele\Lab41-SRI-VOiCES-rm3-tele-mc01-stu-clo.wav');
testTv = zeros(8193*7, 20);
for j=1:20
    testTv(:,j) = reshape(spectrogram(m(j*Fs*30:2:j*Fs*30+Fs*5+1,1)),8193*7,1);
end
testTv = abs(testTv);
testTv = testTv - mean(testTv);
transformedTv = (Transform * testTv)';
tvPredictions = nb.predict(transformedTv);
correctBayes = correctBayes + sum(tvPredictions=="tv");

% repeat for voice
directories = dir('E:\FakeDesktop\voices\VOiCES_devkit\distant-16k\speech\test\rm1\none');
testVoice = zeros(8193*7, 20);
index = 1;
for i=1:length(directories)
    if directories(i).name == "." | directories(i).name == ".."
        continue
    end
    %directories(i).name
    voices = dir(strcat(directories(i).folder, '\', directories(i).name));
    for j=1:2:length(voices)
        if voices(j).name == "." | voices(j).name == ".."
            continue
        end
        [m,Fs] = audioread(strcat(voices(j).folder, '\', voices(j).name));

        testVoice(:,index) = reshape(spectrogram(m(Fs*1:2:Fs*6+1,1)),8193*7,1);
        index = index + 1;
        if index == 21
            break;
        end
    end
    if index == 21
        break;
    end
end
testVoice = abs(testVoice);
testVoice = testVoice - mean(testVoice);
transformedVoice = (Transform * testVoice)';
voicePredictions = nb.predict(transformedVoice);
correctBayes = correctBayes + sum(voicePredictions=="voice");

Accuracy = correctBayes / 80