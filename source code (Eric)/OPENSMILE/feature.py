import os
import opensmile
import pandas as pd
from opensmile import FeatureSet, FeatureLevel

ravdess = 'C:/Users/Eric koh/Desktop/Jupyter/NNDL/Database/Ravdess/'
emodb = 'C:/Users/Eric koh/Desktop/Jupyter/NNDL/Database/emodb/'

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,
    feature_level=opensmile.FeatureLevel.Functionals,
)

ravd_dir = os.listdir(ravdess)
file_emo = []
file_feature = []

for dir in ravd_dir:
    actor = os.listdir(ravdess + dir)
    for file in actor:
        part = file.split('-')
        file_emo.append(int(part[2]))
        file_feature.append(smile.process_file(ravdess + dir + '/' + file))

d = {'Emotion' : file_emo}
emo_df = pd.DataFrame(data=d)
emo_df.Emotion.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
fea_df = pd.concat(file_feature, ignore_index=True)

ravd_df = pd.concat([fea_df, emo_df], axis = 1)
#ravd_df.to_csv('ravdess.csv')

emodb_dir = os.listdir(emodb)
file_emo.clear()
file_feature.clear()
d.clear()

for file in emodb_dir:
    if file[5] == 'W':
        file_emo.append('angry')
    elif file[5] =='L':
        file_emo.append('boredom')
    elif file[5] =='E':
        file_emo.append('disgust')
    elif file[5] =='A':
        file_emo.append('fear')
    elif file[5] =='F':
        file_emo.append('happy')
    elif file[5] =='T':
        file_emo.append('sad')
    elif file[5] =='N':
        file_emo.append('neutral')
    else:
        file_emo.append('unknown')
    file_feature.append(smile.process_file(emodb + file))

d = {'Emotion' : file_emo}
emo_df = pd.DataFrame(data=d)
fea_df = pd.concat(file_feature, ignore_index=True)

emodb_df = pd.concat([fea_df, emo_df], axis = 1)
#emodb_df.to_csv('emodb.csv', index=False)

feature_df = pd.concat([ravd_df, emodb_df])
feature_df.to_csv('feature.csv', index=False)
print(feature_df.shape)