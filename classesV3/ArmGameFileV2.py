from shutil import which
from turtle import colormode
import pandas as pd
import numpy as np
from enum import Enum

# Enumerated class. Allows for conversion of gesture string into gesture value.
class targetClass(Enum):
    rest = 0
    open = 17
    close = 18
    tripod = 38
    pinch = 40


  
# Class responsible for maintaining information about a single trial in a single session
class ArmGameFile:
    def __init__(self,file_path, subject):

        self.armgame_df = pd.read_csv(file_path) # Holds data from ArmGame csv
        self.subject = subject # holds subject code

        self.raw_cm_features = pd.DataFrame() # Holds raw classmeans features from classmeans file
        self.feature_df = pd.DataFrame() # Holds organized features df, a reconstruction of raw_cm_features
        self.f1_distances = pd.DataFrame(columns=range(1,9)) # Holds distances of raw EMG to each f1 EMG avg

        # If armgame_df exists, drop unnecessary columns
        if not self.armgame_df.empty: 
            self.clip_data()

        self.chunk_bounds = {'rest':[],'open':[],'close':[],'pinch':[],'tripod':[]} # Dictionary which will hold tuples of (start,end) for the indexes of each chunk

        self.armgame_df['goal'] = 0 # Add important column to ag_df
        self.armgame_df['total_acc'] = 0 # same

        self.set_goals(['rest','open','close','rest','open','close']) # Construct goal classification order



    # Drops unnecessary columns from armgame_df
    def clip_data(self):
        self.armgame_df = self.armgame_df.drop(columns=['gameName','frameRate','usageLogID','innerZoneRadius','PreRampSpeed','PostRampSpeed','targetClass','userInnerRingRadius','userOuterRingRadius'])



    # Finds chunk bounds, then assigns 'goal' value to each chunk. Goal value represents the gesture the user was tasked to exhibit during trial.
    # 
    # Input: list of goal values, either [0,17,18,0,17,18] for rest->open->close->rest->open->close or [0,17,18,38,40,0,17,18,38,40] with the inclusion of pinch + tripod
    # Output: None, but data is manipulated and armgame_df gains columns with relevant data
    def set_goals(self, goals):
        
        curr = 0
        which_chunk = 0
        df_size = len(self.armgame_df.index)

        # Check each timepoint individually to find the beginning of a chunk (marked by the first instance of a 0 in 'entryType')
        # Increment a counter until 'entryType' is no longer zero. Set 'goal' values equal to passed in goal for the chunk. Record the start and end of the chunk. Append the chunk bounds into self.chunk_bounds.
        while curr < df_size and which_chunk < len(goals):
            if self.armgame_df.at[curr, 'entryType'] == 0: # If current value is a 0, begin looking for next instance of non-0 value. This should only be entered for the first chunk in a file.

                start = curr # Cache starting index

                while curr < df_size and self.armgame_df.at[curr,'entryType'] == 0: # Look for next instance of non-0 value in 'entryType'
                    curr += 1

                self.armgame_df.loc[start:curr, 'goal'] = targetClass[goals[which_chunk]].value # Set all values from index start:curr to the current goal value
                self.chunk_bounds[targetClass[goals[which_chunk]].name].append([start,curr - 1]) # Append chunk boundaries into self.chunk_bounds
                
                which_chunk += 1 # Increment to next chunk
            

            elif self.armgame_df.at[curr,'entryType'] == 3: # If current value is a 3, search for next instance of 0 and then search for next instance of non-zero. 3 values indicate period between trials, which have been included to visualize onset of gesture. 
                start = curr # Cache starting index

                while curr < df_size and self.armgame_df.at[curr, 'entryType'] == 3: # Increment until finding not 3
                    curr += 1

                while curr < df_size and self.armgame_df.at[curr,'entryType'] == 0: # Increment until finding not 0
                    curr += 1


                # Save chunk info same as in earlier if condition
                self.armgame_df.loc[start:curr, 'goal'] = targetClass[goals[which_chunk]].value
                self.chunk_bounds[targetClass[goals[which_chunk]].name].append([start,curr - 1])

                which_chunk += 1 # Increment to next chunk
            
            else: # If 'entryType' does not indicate trial or between trial, increment counter until you find an indicator of trial
                while curr < df_size and self.armgame_df.at[curr,'entryType'] != 0:
                    curr += 1


    # Calculate rolling accuracy of the classifier, adjustable bin size
    # Inputs: chunk boundaries, size of bin output
    # Output: None. armgame_df is manipulated, rolling_acc_bin_binsize column is added.
    def calc_chunk_rolling_acc(self, chunk_start, chunk_end, bin_size):
        
        col_string = 'rolling_acc_bin' + str(bin_size) # Generate column title string based on bin size

        if col_string not in self.armgame_df: # If the column doesn't already exist, instantiate it with 0s 
            self.armgame_df[col_string] = 0
        
        # Create a bin to hold most recent bin_size number of acc values (acc value at given timepoint is a boolean representing whether the gesture was classified as the goal gesture or not. Effect of classifier + subject acc)
        hold = [] 
        for i in range(bin_size): 
            hold.append(0)

        # Iterate over chunk. Replace oldest value in hold with the current value. Divide sum of hold by bin size and save resulting acc percentage to armgame_d[rolling_acc_bin_binsize] at current index.
        for curr in range(chunk_start,chunk_end):
            hold[curr % bin_size] = self.armgame_df.loc[curr,'acc'] 
            self.armgame_df.loc[curr,col_string] = float(sum(hold) / bin_size)


    
    # # Prep for both accuracy calculation functions. Preps data then calls both cumulative acc func and rolling acc func
    def calc_acc(self):
        self.armgame_df['acc'] = np.where(self.armgame_df['class'] == self.armgame_df['goal'], 1, 0) # Set 'acc' equal to the boolean comparison of 'class' and 'goal'
        self.armgame_df['total_chunk_acc'] = 0 # populate cumulative acc column with zeros

        # For each chunk in dataset, calculate accuracy values
        for motion_class in self.chunk_bounds:
            for bounds in self.chunk_bounds[motion_class]:
                start = bounds[0]
                end = bounds[1]
                self.calc_chunk_acc(start,end)
                self.calc_chunk_rolling_acc(start,end,25)



    # Similar to rolling acc, without rolling component. Simply calculates cumulative acc up to current index within chunk.
    def calc_chunk_acc(self, chunk_start, chunk_end):

        i = chunk_start
        sum = 0 # stores total number of correct classifications thus far in chunk

        for val in self.armgame_df.loc[chunk_start:chunk_end,'acc']:
            sum += val # Add current value to sum
            self.armgame_df.loc[i, 'total_chunk_acc'] = sum / (i - chunk_start + 1) # Save sum / current position in chunk 
            i += 1 # Iterate to next index



    # Loads in classmeans file to AG_object
    def load_classmeans_file(self,file_path):
        self.raw_cm_features = pd.read_csv(file_path)
        if len(self.raw_cm_features) < 4:
            self.restructure_cm()
            self.create_feature_df()
            


    def restructure_cm(self):

        self.raw_cm_features = pd.DataFrame(np.vstack([self.raw_cm_features.columns,self.raw_cm_features]))

        self.raw_cm_features.index = ['Rest','Open','Close']

        self.raw_cm_features.loc['Rest'] = self.raw_cm_features.loc['Rest'].astype(float)


    # Interesting method, uses multi-indexing. Special form of dataframe that has hierarchy to certain measures, with some columns having fewer rows. Basically breaks up a df into multiple other dfs and nexts them inside of one another. 
    def create_feature_df(self):

        gestures = [0,17,18] # Set list of relevant gestures
        channels = [*range(1,9)] # Populate list of channel numbers

        iterables = [gestures,channels] # Create an iterables object to pass into MultiIndex.from_product()

        # Essentially, builds a matrix out of all combinations of the iterables passed in and organizes them in hierarchy. 
        index = pd.MultiIndex.from_product(iterables,names=['Gesture','Channel'])

        features = [*range(1,8)] # Populate list of feature numbers

        temp_df = pd.DataFrame(index=index,columns=features) # Build a df using the multiindex as your row index and features as your column index

        # Iterate over channels and populate DF with relevant data
        for i in channels:
            start = 7*(i - 1) + 1 # Get starting point in df for each gesture
            end = start + 7 # get ending point in df for each gesture

            # Populate temp dfs with the feature maps for each gesture
            hold_rest = self.raw_cm_features.iloc[0,start:end] 
            hold_open = self.raw_cm_features.iloc[1,start:end]
            hold_close = self.raw_cm_features.iloc[2,start:end]

            # Populate temp_df with the held feature maps
            temp_df.iloc[i-1,:] = hold_rest
            temp_df.iloc[i + 7,:] = hold_open
            temp_df.iloc[i + 15,:] = hold_close

        # Save multi-indexed df into a member variable
        self.feature_df = temp_df


    # Calculates average distance of current EMG readout to a single raw emg feature (f1_gesture)
    # Input: Relevant gesture
    # Output: dataframe of each channel's current distance from the gestures f1 averages
    def diff_from_f1(self,gesture):

        f1_vals = self.feature_df.loc[(gesture,1):(gesture,8),1].tolist()

        columns = range(1,9)

        temp_df = pd.DataFrame(columns=columns)

        for channel in columns:
            channel_string = 'emgChan' + str(channel)

            temp_df[channel] = [num - float(f1_vals[int(channel)-1]) for num in self.armgame_df[channel_string].tolist()]

        return temp_df