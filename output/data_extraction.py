import os
import opensmile
import pandas as pd
from opensmile import FeatureSet, FeatureLevel

# Each of the 1440 files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 03-01-06-01-02-01-12.wav). These identifiers define the stimulus characteristics:
#
# Filename identifiers
#
# Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
#
# Vocal channel (01 = speech, 02 = song).
#
# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
#
# Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
#
# Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
#
# Repetition (01 = 1st repetition, 02 = 2nd repetition).
#
# Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).


input_dir = 'C:\\Users\\Danson\\Documents\\GitHub\\CZ4042-assignment-grouped\\input'
output_dir = 'C:\\Users\\Danson\\Documents\\GitHub\\CZ4042-assignment-grouped\\output'
config_dir = 'C:\\opensmile-3.0-win-x64\\config\\is09-13\\IS09_emotion.conf'

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,
    feature_level=opensmile.FeatureLevel.Functionals,
)
#y = smile.process_file('audio.wav')

list_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for emotions in list_emotions:
    if not os.path.exists(f'{output_dir}\\{emotions}'):
        os.makedirs(f'{output_dir}\\{emotions}')


list_df_byEmotions = {
    'neutral' : [],
    'calm' : [],
    'happy': [],
    'sad': [],
    'angry': [],
    'fearful': [],
    'disgust': [],
    'surprised': []
}

for folders in os.listdir(input_dir):
    for file in os.listdir(f'{input_dir}\\{folders}'):
        if file[-4:] == '.wav':
            audio_identifiers = file.split('-')
            if audio_identifiers[2] == '01':
                list_df_byEmotions['neutral'].append(smile.process_file(f'{input_dir}\\{folders}\\{file}'))
            elif audio_identifiers[2] == '02':
                list_df_byEmotions['calm'].append(smile.process_file(f'{input_dir}\\{folders}\\{file}'))
            elif audio_identifiers[2] == '03':
                list_df_byEmotions['happy'].append(smile.process_file(f'{input_dir}\\{folders}\\{file}'))
            elif audio_identifiers[2] == '04':
                list_df_byEmotions['sad'].append(smile.process_file(f'{input_dir}\\{folders}\\{file}'))
            elif audio_identifiers[2] == '05':
                list_df_byEmotions['angry'].append(smile.process_file(f'{input_dir}\\{folders}\\{file}'))
            elif audio_identifiers[2] == '06':
                list_df_byEmotions['fearful'].append(smile.process_file(f'{input_dir}\\{folders}\\{file}'))
            elif audio_identifiers[2] == '07':
                list_df_byEmotions['disgust'].append(smile.process_file(f'{input_dir}\\{folders}\\{file}'))
            elif audio_identifiers[2] == '08':
                list_df_byEmotions['surprised'].append(smile.process_file(f'{input_dir}\\{folders}\\{file}'))

df_neutral = list_df_byEmotions['neutral']
df_calm = list_df_byEmotions['calm']
df_happy = list_df_byEmotions['happy']
df_sad = list_df_byEmotions['sad']
df_angry = list_df_byEmotions['angry']
df_fearful = list_df_byEmotions['fearful']
df_disgust = list_df_byEmotions['disgust']
df_surprised = list_df_byEmotions['surprised']

list_emotionDFCollection = [df_neutral, df_calm, df_happy, df_sad, df_angry, df_fearful, df_disgust, df_surprised]

count = 0
for df in list_emotionDFCollection:
    result = pd.concat(df)
    result.to_csv(f'csvs\\{list_emotions[count]}.csv')
    count+=1