import pandas as pd
import numpy as np
import model
import matplotlib.pyplot as plt
import seaborn as sb
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from kerastuner import RandomSearch

print("Reading csv...")
features_df = pd.read_csv('feature.csv')
features_df = features_df[features_df.Emotion != 'calm']
features_df = features_df[features_df.Emotion != 'boredom']
X = features_df.drop('Emotion', axis=1)
Y = features_df['Emotion']

encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.3 , shuffle=True, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
x_val = scaler.fit_transform(x_val)

x_train = np.expand_dims(x_train, axis=2)
x_val = np.expand_dims(x_val, axis=2)
x_test = np.expand_dims(x_test, axis=2)

epoch = 100
batch_size = 32

filepath = "saved_models/weght-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, 
                              monitor='val_acc', 
                              verbose=1, 
                              save_best_only=True, 
                              mode='max')

# Log epoch, acc, loss, val_acc, val_loss
log_csv = CSVLogger('logs_testrun1.csv',
                    separator=',', 
                    append=True)

callback_list = [checkpoint, log_csv]

# HP tuner search
tuner = RandomSearch(hypermodel = model.sm_model,
                     objective = "val_acc",
                     max_trials = 6,
                     executions_per_trial = 1,
                     project_name = "Trials")
tuner.search_space_summary()
tuner.search(x_train,
             y_train,
             epochs = 50,
             validation_data =(x_val, y_val),
             verbose = 2
)

"""
model = model.sm_model(x_train.shape[1])
history = model.fit(x_train, y_train, 
                    validation_data=(x_val, y_val), 
                    batch_size=batch_size,
                    epochs=epoch,
                    verbose=2,
                    callbacks = callback_list)

print("Accuracy of our model on test data : " , model.evaluate(x_test,y_test, verbose=2)[1]*100 , "%")

fig , ax = plt.subplots(1,2)
train_acc = history.history['acc']
train_loss = history.history['loss']
test_acc = history.history['val_acc']
test_loss = history.history['val_loss']

fig.set_size_inches(20,6)
ax[0].plot(train_loss, label = 'Training Loss')
ax[0].plot(test_loss , label = 'Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(train_acc, label = 'Training Accuracy')
ax[1].plot(test_acc , label = 'Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.savefig('sm_model_plot.png')

pred_test = model.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)

y_test_class = encoder.inverse_transform(y_test)
cm = confusion_matrix(y_test_class, y_pred)
plt.figure(figsize = (12, 10))
cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
sb.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.savefig("cm.png")

print(classification_report(y_test_class, y_pred))
"""