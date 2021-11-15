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

input_dir = '\\GitHub\\CZ4042-assignment-grouped\\input'
output_dir = '\\GitHub\\CZ4042-assignment-grouped\\output'
config_dir = '\\opensmile-3.0-win-x64\\config\\is09-13\\IS09_emotion.conf'

## list of opensmile featuresets
#ComParE_2016
#GeMAPS
#GeMAPSv01a
#GeMAPSv01b
#eGeMAPS
#eGeMAPSv01a
#eGeMAPSv01b
#eGeMAPSv02
#emobase

deprecated_featuresets = ['GeMAPS', 'eGeMAPSv01b', 'eGeMAPS']
list_opensmile_featureSets = [
    [opensmile.FeatureSet.ComParE_2016, 'ComParE_2016'],
    [opensmile.FeatureSet.GeMAPS, 'GeMAPS'],
    [opensmile.FeatureSet.GeMAPSv01a, 'GeMAPSv01a'],
    [opensmile.FeatureSet.GeMAPSv01b, 'GeMAPSv01b'],
    [opensmile.FeatureSet.eGeMAPS, 'eGeMAPS'],
    [opensmile.FeatureSet.eGeMAPSv01a, 'eGeMAPSv01a'],
    [opensmile.FeatureSet.eGeMAPSv01b, 'eGeMAPSv01b'],
    [opensmile.FeatureSet.eGeMAPSv02, 'eGeMAPSv02'],
    [opensmile.FeatureSet.emobase, 'emobase'],
]

for featureset in list_opensmile_featureSets:
    print(f'Currently extracting features with the "{[featureset[1]]}" featureset.')
    smile = opensmile.Smile(
        feature_set=featureset[0],
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    list_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised', 'boredom']

    list_df_byEmotions = {
        #ravdess emotions list
        'neutral' : [],
        'calm' : [],
        'happy': [],
        'sad': [],
        'angry': [],
        'fearful': [],
        'disgust': [],
        'surprised': [],

        #emodb emotions list - exclusive of overlapping emotions like: angry, disgust, fearful, happy, neutral and sadness
        'boredom' : []
    }


    for folders in os.listdir(input_dir):
        if folders != 'emodb':
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
        else:
            for file in os.listdir(f'{input_dir}\\{folders}'):
                audio_identifiers = file[5]
                if audio_identifiers == 'W':
                    list_df_byEmotions['angry'].append(smile.process_file(f'{input_dir}\\{folders}\\{file}'))
                elif audio_identifiers == 'L':
                    list_df_byEmotions['boredom'].append(smile.process_file(f'{input_dir}\\{folders}\\{file}'))
                elif audio_identifiers == 'E':
                    list_df_byEmotions['disgust'].append(smile.process_file(f'{input_dir}\\{folders}\\{file}'))
                elif audio_identifiers == 'A':
                    list_df_byEmotions['fearful'].append(smile.process_file(f'{input_dir}\\{folders}\\{file}'))
                elif audio_identifiers == 'F':
                    list_df_byEmotions['happy'].append(smile.process_file(f'{input_dir}\\{folders}\\{file}'))
                elif audio_identifiers == 'T':
                    list_df_byEmotions['sad'].append(smile.process_file(f'{input_dir}\\{folders}\\{file}'))
                elif audio_identifiers == 'N':
                    list_df_byEmotions['neutral'].append(smile.process_file(f'{input_dir}\\{folders}\\{file}'))

    df_neutral = list_df_byEmotions['neutral']
    df_calm = list_df_byEmotions['calm']
    df_happy = list_df_byEmotions['happy']
    df_sad = list_df_byEmotions['sad']
    df_angry = list_df_byEmotions['angry']
    df_fearful = list_df_byEmotions['fearful']
    df_disgust = list_df_byEmotions['disgust']
    df_surprised = list_df_byEmotions['surprised']
    df_boredom = list_df_byEmotions['boredom']

    list_emotionDFCollection = [df_neutral, df_calm, df_happy, df_sad, df_angry, df_fearful, df_disgust, df_surprised, df_boredom]


    print(f'Generating csv file for featureset: {featureset[1]}')
    count = 0
    for features in list_opensmile_featureSets:
        df_featureset = pd.DataFrame()
        emotion_index = 0
        for list_df_emotion in list_emotionDFCollection:
            number_of_rows = len(list_df_emotion)
            labels = [f'{list_emotions[emotion_index]}' for x in range(number_of_rows)]
            df_emotion = pd.concat(list_df_emotion)
            df_emotion['label'] = labels
            if emotion_index == 0:
                df_featureset = df_emotion
            else:
                df_featureset = pd.concat([df_featureset, df_emotion])
            emotion_index += 1
        if featureset[1] in deprecated_featuresets:
            df_featureset.to_csv(f'{output_dir}\\{featureset[1]} (Deprecated).csv')
        else:
            df_featureset.to_csv(f'{output_dir}\\{featureset[1]}.csv')
        count += 1