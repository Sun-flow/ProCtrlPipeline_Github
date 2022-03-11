from shutil import which
from turtle import colormode
import pandas as pd
import numpy as np
from enum import Enum

class targetClass(Enum):
    rest = 0
    open = 17
    close = 18
    tripod = 38
    pinch = 40


class ArmGameFile:
    def __init__(self,file_path, subject):

        self.armgame_df = pd.read_csv(file_path)
        self.subject = subject

        self.raw_cm_features = pd.DataFrame()
        self.feature_df = pd.DataFrame()
        self.f1_distances = pd.DataFrame(columns=range(1,9))

        if not self.armgame_df.empty:
            self.clip_data()

        self.chunk_bounds = {'rest':[],'open':[],'close':[],'pinch':[],'tripod':[]}

        self.armgame_df['goal'] = 0
        self.armgame_df['total_acc'] = 0

        self.set_goals(['rest','open','close','rest','open','close'])


    def clip_data(self):
        self.armgame_df = self.armgame_df.drop(columns=['gameName','frameRate','usageLogID','innerZoneRadius','PreRampSpeed','PostRampSpeed','targetClass','userInnerRingRadius','userOuterRingRadius'])


    def set_goals(self, goals):
        
        curr = 0
        which_chunk = 0
        df_size = len(self.armgame_df.index)

        while curr < df_size and which_chunk < len(goals):
            if self.armgame_df.at[curr, 'entryType'] == 0:

                start = curr 

                while curr < df_size and self.armgame_df.at[curr,'entryType'] == 0:
                    curr += 1

                self.armgame_df.loc[start:curr, 'goal'] = targetClass[goals[which_chunk]].value
                self.chunk_bounds[targetClass[goals[which_chunk]].name].append([start,curr - 1])
                
                which_chunk += 1
            
            elif self.armgame_df.at[curr,'entryType'] == 3:
                start = curr

                while curr < df_size and self.armgame_df.at[curr, 'entryType'] == 3:
                    curr += 1

                while curr < df_size and self.armgame_df.at[curr,'entryType'] == 0:
                    curr += 1

                self.armgame_df.loc[start:curr, 'goal'] = targetClass[goals[which_chunk]].value
                self.chunk_bounds[targetClass[goals[which_chunk]].name].append([start,curr - 1])

                which_chunk += 1
            
            else:
                while curr < df_size and self.armgame_df.at[curr,'entryType'] != 0:
                    curr += 1


    def calc_chunk_rolling_acc(self, chunk_start, chunk_end, bin_size):
        
        col_string = 'rolling_acc_bin' + str(bin_size)

        if col_string not in self.armgame_df:
            self.armgame_df[col_string] = 0
        
        hold = []
        for i in range(bin_size):
            hold.append(0)

        for curr in range(chunk_start,chunk_end):
            hold[curr % bin_size] = self.armgame_df.loc[curr,'acc'] 
            self.armgame_df.loc[curr,col_string] = float(sum(hold) / bin_size)



    def calc_acc(self):
        self.armgame_df['acc'] = np.where(self.armgame_df['class'] == self.armgame_df['goal'], 1, 0)
        self.armgame_df['total_chunk_acc'] = 0

        for motion_class in self.chunk_bounds:
            for bounds in self.chunk_bounds[motion_class]:
                start = bounds[0]
                end = bounds[1]
                self.calc_chunk_acc(start,end)
                self.calc_chunk_rolling_acc(start,end,25)

    
    def calc_chunk_acc(self, chunk_start, chunk_end):

        i = chunk_start
        sum = 0

        for val in self.armgame_df.loc[chunk_start:chunk_end,'acc']:
            sum += val 
            self.armgame_df.loc[i, 'total_chunk_acc'] = sum / (i - chunk_start + 1)
            i += 1


    def load_classmeans_file(self,file_path):
        self.raw_cm_features = pd.read_csv(file_path)
        if len(self.raw_cm_features) < 4:
            self.restructure_cm()
            self.create_feature_df()
            

    def restructure_cm(self):

        self.raw_cm_features = pd.DataFrame(np.vstack([self.raw_cm_features.columns,self.raw_cm_features]))

        self.raw_cm_features.index = ['Rest','Open','Close']

        self.raw_cm_features.loc['Rest'] = self.raw_cm_features.loc['Rest'].astype(float)


    def create_feature_df(self):

        gestures = [0,17,18]
        channels = [*range(1,9)]

        iterables = [gestures,channels]

        index = pd.MultiIndex.from_product(iterables,names=['Gesture','Channel'])

        features = [*range(1,8)]

        temp_df = pd.DataFrame(index=index,columns=features)

        for i in channels:
            start = 7*(i - 1) + 1
            end = start + 7

            hold_rest = self.raw_cm_features.iloc[0,start:end]
            hold_open = self.raw_cm_features.iloc[1,start:end]
            hold_close = self.raw_cm_features.iloc[2,start:end]
            temp_df.iloc[i-1,:] = hold_rest
            temp_df.iloc[i + 7,:] = hold_open
            temp_df.iloc[i + 15,:] = hold_close

        self.feature_df = temp_df


    def diff_from_f1(self,gesture):

        f1_vals = self.feature_df.loc[(gesture,1):(gesture,8),1].tolist()

        columns = range(1,9)

        temp_df = pd.DataFrame(columns=columns)

        for channel in columns:
            channel_string = 'emgChan' + str(channel)

            temp_df[channel] = [num - float(f1_vals[int(channel)-1]) for num in self.armgame_df[channel_string].tolist()]

        return temp_df