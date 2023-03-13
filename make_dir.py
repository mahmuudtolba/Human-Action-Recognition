import mediapipe as mp
import numpy as np
import os

mp_holistic_model = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 

from processing_utils import make_dir
#that is the path for exported data
actions = np.array(['open' , 'put' , 'close' , 'stand'])

DATA_PATH = 'mediapipe_data'
for action in actions:
        for sequence in range(250):
            try:
                os.makedirs(os.path.join(DATA_PATH , action , str(sequence)))
            except Exception as e:
                print(f'we have {e}')
        print(f'{os.path.join(DATA_PATH , action )} is created')