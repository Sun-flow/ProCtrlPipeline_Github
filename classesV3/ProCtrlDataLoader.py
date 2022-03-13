import pandas as pd
import numpy as np
import os 
import pprint

import matplotlib.pyplot as plt

from classesV3.ArmGameFileV2 import ArmGameFile

from sklearn.metrics import confusion_matrix

import seaborn as sn


# Loads in data for all subjects, focuses on overarching structures. High ties to ArmGameV2, which holds subject+session+trial specific data
class ProCtrlDataLoader:
    def __init__(self, data_path, input_training_groups, input_subjs, input_sessions):

        self.data_path = data_path

        self.training_groups = dict.fromkeys(input_training_groups)

        self.subjects_of_interest = input_subjs

        self.sessions_of_interest = input_sessions

        if not self.sessions_of_interest:
            self.sessions_of_interest = ['sess2','sess3','sess4','sess5','sess6']

        self.confusion_matrix_df = pd.DataFrame(columns = ['SUBJ','GROUP','SESSION','VAL1','VAL2','VAL3','VAL4','VAL5','VAL6','VAL7','VAL8','VAL9'])

    
    # Creates a dictionary (self.training_groups) which preserves all file path information [group][subject][session][pre/post_train] = AG_object
    def loaddata(self):

        # Get all file paths below data_path directory
        for root,dirs,files in os.walk(self.data_path, topdown = False):

            # For each file found, do:
            for file in files:

                # Search for string 'ArmGame' in file name. If found, do: 
                if 'ArmGame' in file:

                    # Splits file string into relevant info, based on known relationships
                    file_path = os.path.join(root,file)
                    split_file_path = file_path.split('/')
                    file_group = split_file_path[len(split_file_path) - 5]
                    subject_code = split_file_path[len(split_file_path) - 4]
                    session = split_file_path[len(split_file_path) - 3]
                    pre_post = split_file_path[len(split_file_path) - 2]
                    file_name = split_file_path[len(split_file_path) - 1]
                    
                    # If the 'group' of this file is one of those selected to load (passed in to constructor), do:
                    if file_group in self.training_groups.keys():
                        
                        # If the current subject is not in the given list of relevant subjects, skip to next run of loop. otherwise, do:
                        if self.subjects_of_interest and subject_code not in self.subjects_of_interest:
                            continue
                        else:

                            # If current session is in list of relevant sessions
                            if session in self.sessions_of_interest:
                                
                                #  Create an ArmGameFile object using this current file & subject
                                ag_file = ArmGameFile(file_path,subject_code)

                                # Check to see if 'classmeans' is in this subject,sessions,prepost's directory. If so, load it's data into AG_object
                                for root,dir,files in os.walk(root,topdown=False):
                                    for file in files:
                                        if 'ClassMeans' in file:
                                            classmeans_fp = os.path.join(root,file)
                                            ag_file.load_classmeans_file(classmeans_fp)


                                # The following conditions set up dictionary with all relevant data.
                                # End format is self.training_groups[group][subject][session][pre/post_train] = AG_file

                                #######
                                # Basic method is:
                                # If training group does NOT already exist in dictionary:
                                #   Add {Group -> {Subject -> {Session -> {Pre/post = AG_object}}}}
                                # If training group exists but Subject DOES NOT exist in training group:
                                #   Add {Subject -> {Session -> {Pre/Post = AG_object}}} to training group
                                # If training group & subject exists but session DOES NOT exist:
                                #   Add {Session -> {Pre/Post = AG_object}} to Subject
                                # If all of the above but not pre/post exists:
                                #   Add {Pre/Post = AG_object} to Session 
                                ########

                                if not self.training_groups[file_group]:
                                    self.training_groups[file_group] = {subject_code:{session:{pre_post:ag_file}}}
                            
                                if subject_code not in self.training_groups[file_group]:
                                    self.training_groups[file_group][subject_code] = {session:{pre_post:ag_file}}

                                if session not in self.training_groups[file_group][subject_code]:
                                    self.training_groups[file_group][subject_code][session] = {pre_post:ag_file}

                                if pre_post not in self.training_groups[file_group][subject_code][session]:
                                    self.training_groups[file_group][subject_code][session][pre_post] = ag_file

                                

    # Creates a Dataframe that includes all confusion matrix values for each subject and set of gestures
    # This is an aggregate df, meaning it spans all subjects that have been loaded by current ProCtrlDataLoader object 
    def aggregate_sess_x_group_df(self,which_groups):
        
        # For Every group in passed in selection of groups
        for group in which_groups:
            print(group)

            # For each participant(key) in that group
            for participant in self.training_groups[group].keys():

                # For each session(key) in that participant
                for sess in self.training_groups[group][participant].keys():

                    # Run member function .confuse() on relevant tags, returns cm value for each gesture pair
                    cm = self.confuse(group,participant,sess,False)

                    # Begin constructing row, starting with tags
                    row = [participant, group, sess]

                    # Append each item passed out of .confuse() to the previously generated row
                    # Output: [group,subj,sess,cm1,cm2,cm3...cm9] 
                    for item in [*cm]:
                        row.extend(item)
                    
                    # Attach row to growing cm dataframe at next index
                    self.confusion_matrix_df.loc[len(self.confusion_matrix_df)] = row

        # Once the whole cm dataframe is populated, sort it by subject and session (will naturally sort by group, as subjects are unique to group)
        self.confusion_matrix_df = self.confusion_matrix_df.sort_values(by=["SUBJ","SESSION"])

        # Save df to output file
        self.confusion_matrix_df.to_csv('aggregated_confusion_matrix_df.csv', index = False)

        # Also returns cm_df, in case I want to do something with it externally
        return self.confusion_matrix_df



    # Create a confusion matrix for gestures of a particular subject & session (includes both pre & post, if applicable)
    def confuse(self, group, subj, sess, figBOOL):

        # Create temp df to hold data in
        hold = pd.DataFrame()

        # Populate hold by concatenating pre and post dfs. Includes all active trial timepoints for the entire session. 
        for pre_post in self.training_groups[group][subj][sess].keys():
            if 'trained' in pre_post:
                hold = pd.concat([hold, self.training_groups[group][subj][sess][pre_post].armgame_df])

        # Run sklearn's confusion matrix function on hold, comparing classifier output to goal output
        cm = confusion_matrix(hold['class'], hold['goal'])

        # Convert data into readable numbers
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Return dataframe of accuracy values (3x3 or 5x5 grid)
        return cmn



    # Print out dictionary, all of its key + value pairs. 
    # This is a depreciated function that I used to make sure I was creating my dictionaries properly. No longer of need, yet it persists in the code, waiting, for some such lonely night that I call to it, once again, to print for me.
    def printDict(self):
        pp = pprint.PrettyPrinter(depth=4)
        for group in self.training_groups:
            pp.pprint(group)
            print(*self.training_groups[group].keys())



    # This function is used to output accuracy graphs over time for many armgame trials. At one point, it functioned to print cumulative accuracy up to current timepoint. Now it maintains cumulative accuracy for only the most recent 25 datapoints.
    # In english:
    #   any timepoint within trial t will be represented on the x axis
    #   cumulative classification accuracy over most recent 25 timepoints on y axis
    #   f(t) = sum(acc(t) + acc(t - 1) + acc(t-2) + ... + acc(t-24)) / 25 
    #   Each line represents one trial
    # 
    # Inputs: [relevant groups],[relevant subjects], [relevant sessions], [relevant motion classes]
    # Outputs: Acc graph aggregating over passed in tags (displays timeline of rolling acc for all runs that meet passed in criteria). Currently not designed to aggregate across subjects, only within. Will output one graph per subject passed in, which includes acc lines for all session and gesture tags passed in.
    # 
    # This generality is very useful for me, as I only need to know what trial tags im looking for and the rest of the data is preloaded. So I don't have to prep or pass in any data, I just tell it what data to utilize, bc it's already present in the DataLoader object.
    # Ideally I would update this function to be a bit more streamlined, but I'm not currently using it for much and it works fine as is. Could be simplified easily nontheless. 
    def print_acc_lines(self, groups, subjects, sessions, motion_classes, title):

        i = 0
        # raw_acc_df = pd.DataFrame() # Depreciated line, maintained in case I ever make this function work for cumulative acc again
        # rolling_acc_df = pd.DataFrame()

        chunks = []

        # Get group tag
        for group in groups:

            # Get relevant subject tags, unless none was passed in, then do all of the subjects in that group
            curr_subjs = subjects
            if not subjects:
                curr_subjs = self.training_groups[group].keys()
            
            # Process each subject of interest
            for subject in curr_subjs:
                
                # Create temporary dataframe to hold just this subject's accuracy values. Later on, this will be appended to the subject's ArmGame_df which holds all of the trial data.
                rolling_acc_df = pd.DataFrame()
                i = 0

                # Process each session of interest for this subject
                for session in sessions:

                    # If only certain gestures are of interest, select for them. If none were passed in, do all of them. Same as earlier for curr_subjs
                    #   This method is more confusing in code, but more simple when calling the function. Bc of this implementation, I can call a function as such:
                    #   print_acc_lines(['bio','arb'],[],[],['rest','open'], title)   # A call to func
                    #   print_acc_lines(groups,subjects,sessions,gestures,title)      # Default constructor for reference
                    #   This call would output acc graphs for only rest and open (excluding close, pinch, tripod), for every subject and every session in the bio or arb training group
                    curr_classes = motion_classes
                    if not motion_classes:
                        curr_classes = ['rest','open','close']

                    # Process each gesture of interest
                    for motion_class in curr_classes:

                        # Process each trial in the session
                        for pre_post in self.training_groups[group][subject][session].keys():
                            
                            # Create placeholder var to current subj_sess_trial so I don't have to index repeatedly
                            AG_object = self.training_groups[group][subject][session][pre_post]

                            # Calls function that calculates the classifiers accuracy at each time point (assigns a bool value to a column representing whether or not classification == goal). This column is saved in armgame_df.
                            # Also finds cumulative accuracy for both whole trial and last 25 timepoints of trial 
                            AG_object.calc_acc()                            
                            
                            # Data is chunked within AG_object based on active trial. Each run contains 6 chunks, with goal class order: rest -> open -> close -> rest -> open -> close. AG object contains a dict called chunk_bounds, where keys = {'rest','open','close'} and each key maps to an array of tuples for the bounds of each trial.
                            #
                            # So AG_object.chunk_bounds['rest'] would return the list [(start1, end1),(start2, end2)] bc there are two chunks labeled rest, each with its own bounds
                            for bounds in AG_object.chunk_bounds[motion_class]:

                                # Depreciated lines which get cumulative accuracy data for chunk
                                data = AG_object.armgame_df.loc[bounds[0]:bounds[1], 'total_chunk_acc']
                                data = data.reset_index(drop=True)

                                # Gets rolling accuracy data out of armgame_df
                                roll_data = AG_object.armgame_df.loc[bounds[0]:bounds[1], 'rolling_acc_bin25']
                                roll_data = roll_data.reset_index(drop=True)

                                # Create plot title out of relevant tags
                                column_title = subject + '_' + session + '_' + motion_class + '_' + pre_post + str(i)

                                # Inserts line into a dataframe which is storing all lines to be printed
                                #raw_acc_df.insert(loc=i,column=i, value=data)
                                rolling_acc_df.insert(loc=i,column=column_title, value=roll_data)
                                i += 1


                # Makes a title which includes whatever was passed into function as title and subject code
                per_subj_title = title + subject
                
                # Plots acc lines
                per_subj_lines = rolling_acc_df.plot.line(title = per_subj_title)
                
                # Saves fig
                per_subj_lines.figure.savefig('./figs/acc_line_plts/' + per_subj_title)

                # Returns the plot, potentially useful if I want to used the plot externally to the function. Depreciated, bc when I wrote this I didn't really understand plots. In the future I would update this to pass out the fig object, instead of the plot object. 
                return per_subj_lines

        #hold = raw_acc_df.plot.line(title = title)

        # hold = rolling_acc_df.plot.line(title = title)

        # hold.legend(fontsize=7)
        # hold.figure.savefig('./figs/acc_line_plts/' + title)
