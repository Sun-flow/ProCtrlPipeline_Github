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


                                # Following conditions set up dictionary with all relevant data.

                                # End format is self.training_groups[group][subject][session][pre/post_train] = AG_file
                                if not self.training_groups[file_group]:
                                    self.training_groups[file_group] = {subject_code:{session:{pre_post:ag_file}}}
                            
                                if subject_code not in self.training_groups[file_group]:
                                    self.training_groups[file_group][subject_code] = {session:{pre_post:ag_file}}

                                if session not in self.training_groups[file_group][subject_code]:
                                    self.training_groups[file_group][subject_code][session] = {pre_post:ag_file}

                                if pre_post not in self.training_groups[file_group][subject_code][session]:
                                    self.training_groups[file_group][subject_code][session][pre_post] = ag_file

                                

    # Creates a Dataframe that includes all confusion matrix values for each subject and set of gestures
    def aggregate_sess_x_group_df(self,which_groups):
        for group in which_groups:
            print(group)
            for participant in self.training_groups[group].keys():
                for sess in self.training_groups[group][participant].keys():
                    cm = self.confuse(group,participant,sess,False)

                    row = [participant, group, sess]

                    for item in [*cm]:
                        row.extend(item)

                    self.confusion_matrix_df.loc[len(self.confusion_matrix_df)] = row

        self.confusion_matrix_df = self.confusion_matrix_df.sort_values(by=["SUBJ","SESSION"])
        self.confusion_matrix_df.to_csv('aggregated_confusion_matrix_df.csv', index = False)

        return self.confusion_matrix_df

    # Create a confusion matrix for gestures of a particular subject & session (includes both pre & post, if applicable)
    def confuse(self, group, subj, sess, figBOOL):
        hold = pd.DataFrame()

        for pre_post in self.training_groups[group][subj][sess].keys():
            if 'trained' in pre_post:
                hold = pd.concat([hold, self.training_groups[group][subj][sess][pre_post].armgame_df])

        cm = confusion_matrix(hold['class'], hold['goal'])
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        return cmn


    def printDict(self):
        pp = pprint.PrettyPrinter(depth=4)
        for group in self.training_groups:
            pp.pprint(group)
            print(*self.training_groups[group].keys())


    def print_acc_lines(self, groups, subjects, sessions, motion_classes, title):

        i = 0
        raw_acc_df = pd.DataFrame()
        # rolling_acc_df = pd.DataFrame()

        chunks = []

        for group in groups:
            curr_subjs = subjects
            if not subjects:
                curr_subjs = self.training_groups[group].keys()
            
            for subject in curr_subjs:

                rolling_acc_df = pd.DataFrame()
                i = 0

                for session in sessions:
                    curr_classes = motion_classes
                    if not motion_classes:
                        curr_classes = ['rest','open','close']

                    for motion_class in curr_classes:

                        for pre_post in self.training_groups[group][subject][session].keys():
                            AG_object = self.training_groups[group][subject][session][pre_post]

                            AG_object.calc_acc()                            
                            
                            for bounds in AG_object.chunk_bounds[motion_class]:

                                data = AG_object.armgame_df.loc[bounds[0]:bounds[1], 'total_chunk_acc']
                                data = data.reset_index(drop=True)

                                roll_data = AG_object.armgame_df.loc[bounds[0]:bounds[1], 'rolling_acc_bin25']
                                roll_data = roll_data.reset_index(drop=True)

                                column_title = subject + '_' + session + '_' + motion_class + '_' + pre_post + str(i)

                                #raw_acc_df.insert(loc=i,column=i, value=data)
                                rolling_acc_df.insert(loc=i,column=column_title, value=roll_data)
                                i += 1

                per_subj_title = title + subject

                per_subj_lines = rolling_acc_df.plot.line(title = per_subj_title)
                
                per_subj_lines.figure.savefig('./figs/acc_line_plts/' + per_subj_title)

                return per_subj_lines

        #hold = raw_acc_df.plot.line(title = title)

        # hold = rolling_acc_df.plot.line(title = title)

        # hold.legend(fontsize=7)
        # hold.figure.savefig('./figs/acc_line_plts/' + title)
