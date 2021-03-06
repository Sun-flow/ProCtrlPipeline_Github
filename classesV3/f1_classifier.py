import pandas as pd

# Gets distance of each timepoint's raw EMG to the avg f1 of each gesture. Then, performs classification based on which of the distances are shortest. Most of the loop in this function serves to remove non-trial datapoints.
# Inputs: AG_object
# Output: List of List of distances for each gesture at each time point.
def classify_raw_emg_over_f1(ag_object):

    # Get Diff from f1 for each gesture at each timepoint, save resulting dfs to temp dfs
    dist_to_rest = ag_object.diff_from_f1(0)
    dist_to_open = ag_object.diff_from_f1(17)
    dist_to_close = ag_object.diff_from_f1(18)


    f1_distance_avgs = pd.DataFrame(columns = ['0','17','18','f1_dist_class','coapt_class', 'goal_output']) # Construct empty df with needed columns

    # Get gesture keys to consider
    for gesture in ag_object.chunk_bounds:

        if gesture == 'pinch' or gesture == 'tripod': # Skip currently irrelevant gesture
            continue # Continue just ends the current iteration of loop and begins the next one. Like break or pass. 
        for bounds in ag_object.chunk_bounds[gesture]: # check both runs in current gesture
        
            temp_chunk_dist_avgs = pd.DataFrame(columns = ['0','17','18'])
            
            start = bounds[0]
            end = bounds[1]

            curr_df = ag_object.armgame_df # make temp var to hold ag_object (easier to reference)
            
            # Save relevant distance averages to their corresponding gesture
            temp_chunk_dist_avgs['0'] = abs(dist_to_rest.iloc[start:end].sum(axis=1) / 8)
            temp_chunk_dist_avgs['17'] = abs(dist_to_open.iloc[start:end].sum(axis=1) / 8)
            temp_chunk_dist_avgs['18'] = abs(dist_to_close.iloc[start:end].sum(axis=1) / 8)
            

            temp_chunk_dist_avgs['f1_dist_class'] = temp_chunk_dist_avgs.idxmin(axis='columns') # Set f1 distance classification to be equal to the column title containing the lowest value in that row (if distance vals are '0' = 10, '17' = 22, '18' = 8, then classifier would output 18)

            temp_chunk_dist_avgs['coapt_class'] = curr_df['class'].iloc[start:end] # Set coapt classification to whatever it classified in the armgame_df

            temp_chunk_dist_avgs['goal_output'] = curr_df['goal'].iloc[start:end] # Set goal output to whatever it was in armgame_df
            # temp_chunk_dist_avgs = temp_chunk_dist_avgs.reset_index(drop = True)        

            # print(temp_chunk_dist_avgs.to_string())

            f1_distance_avgs = f1_distance_avgs.append(temp_chunk_dist_avgs) # Append values for current chunk to dataframe holding values for all chunks

    return f1_distance_avgs


# Determines accuracy for each classifier, as well as the agreement between classifiers
def get_accuracy_for_f1_classifier(f1_classifier_df):
    temp_df = pd.DataFrame(columns=['f1_class_acc','coapt_class_acc', 'f1_coapt_agreement']) # Build temp df with necessary columns

    # Populate columns of temp_df by comparing values from the classifier_df produced in the last function. places 1s where vals in compared columns are ==, and 0 where !=
    temp_df['f1_class_acc'] = pd.to_numeric(f1_classifier_df['f1_dist_class']) & f1_classifier_df['goal_output'] # Set f1 class acc to boolean assertion of goal output & f1_class
    temp_df['coapt_class_acc'] = f1_classifier_df['coapt_class'] & pd.to_numeric(f1_classifier_df['goal_output']) # Set coapt class acc to boolean assertion of goal output & coapt_class
    temp_df['f1_coapt_agreement'] = pd.to_numeric(f1_classifier_df['f1_dist_class']) & f1_classifier_df['coapt_class'] # Set agreement between classifiers to assertion of f1_class & coapt_class

    # Find acc percentages by summing columns & dividing by length of column. Save outputs into holder vars.
    f1_class_acc = sum(temp_df['f1_class_acc']) / len(temp_df)
    coapt_class_acc = sum(temp_df['coapt_class_acc']) / len(temp_df)
    f1_coapt_agreement = sum(temp_df['f1_coapt_agreement']) / len(temp_df)

    # Build list of accuracies to be returned. 
    temp_accuracies = [f1_class_acc,coapt_class_acc,f1_coapt_agreement]

    return temp_df, temp_accuracies


# Check the agreement between the f1 classifier and the coapt classifier
def dataset_f1_agreement(DataSet):

    temp_df = pd.DataFrame(columns=['group','subj','sess','f1_class_acc','coapt_class_acc','f1_coapt_agreement']) # Build temp df with proper column names

    for group in DataSet.data_dict: # Run each group in loaded DataSet
        for participant in DataSet.data_dict[group].keys(): # Run each participant in current group
            if 'bi05' in participant: # Skip this one problem participant (bandaid solution, they are missing data)
                continue
            for sess in DataSet.data_dict[group][participant].keys(): # Run each session for this participant
                temp_acc_values = [0,0,0]
                i = 0
                for prepost in ['pre_trained','post_trained']: # Run each trial in this session. I am averaging within session, so this loop is responsible for getting data from each trial and avging it.
                    if prepost not in DataSet.data_dict[group][participant][sess].keys(): # If one of the trials is not present, skip
                        continue
                    temp_ag_object = DataSet.data_dict[group][participant][sess][prepost] # Get AG_object
                    temp_f1_classifier = classify_raw_emg_over_f1(temp_ag_object) # Run classifier over data
                    junk_df,hold_acc_values = get_accuracy_for_f1_classifier(temp_f1_classifier) # Get classifier accuracy

                    temp_acc_values = [a + b for a, b in zip(temp_acc_values,hold_acc_values)]  # Add found acc values to prepped list
                    i += 1
                
                temp_acc_values = [x / i for x in temp_acc_values] # When all trials in this sess have been added, divide by number of trials
                new_row = [group,participant,sess,*temp_acc_values] # Create new row to append to temp_df with values found for this subj+sess
                temp_df.loc[len(temp_df)] = new_row # Append new row to temp_df


    return temp_df # Return dataframe of acc / agreement values 
