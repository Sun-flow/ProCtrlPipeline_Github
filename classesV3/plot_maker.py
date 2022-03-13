import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import gridspec
from IPython import display
import numpy as np

def make_worm(data, group,subj,sess,gesture,prepost, which_worm):
    armgame_object = data.training_groups[group][subj][sess][prepost]
    armgame_object.calc_acc()
    #armgame_df = armgame_object.df
    ag_bounds = armgame_object.chunk_bounds[gesture][0]

    feature1 = []

    if gesture == 'rest':
        feature1 = armgame_object.feature_df.loc[(0,1):(0,8),1]
    elif gesture == 'open':
        feature1 = armgame_object.feature_df.loc[(17,1):(17,8),1].tolist()
        print(feature1)
    elif gesture == 'close':
        feature1 = armgame_object.feature_df.loc[(18,1):(18,8),1]

    title = subj + '_' + sess + '_' + gesture + '_' + prepost + '_worm'

    radar_gif(armgame_object, feature1, ag_bounds, title, which_worm)



# Sets up a radar plot for later plotting.
# Inputs: figure object, 'categories' or axis that are being plotted on (in this case, basically always emg channels for me), gridspec values (values which determine where the radarplot exists in a larger plot, default to [1,1,1])
# Outputs: radar ax object (subplot object, can be manipulated later), ANGLES list (includes some relevant shape data about plot) 
def radar_plot_setup(fig,categories,gridspec=[1,1,1]):

    # Determines number of axis to plot over
    spokes = len(categories)

    # Get angular degree locations for each axis
    ANGLES = [n / spokes * 2 * np.pi for n in range(spokes)]
    # Due to the way plotting works, it's necessary to repeat the first axis at the end. This will also be done with the first data point plotted on the axis. 
    ANGLES += ANGLES[:1]

    # Create the radar plot as a subplot of the fig object passed in
    polar_ax = fig.add_subplot(gridspec, polar=True)

    # Set some relevant parameters for shape
    polar_ax.set_theta_offset(np.pi / 2)
    polar_ax.set_theta_direction(-1)

    polar_ax.set_xticks(ANGLES[:-1])
    polar_ax.set_xticklabels(categories, size=14)

    # Return the ax object, which will be mutable and can have elements added or removed from it. Also return ANGLES, as it will stay relevant for plotting
    return polar_ax, ANGLES



# Same as above, but for a line graph. Much simpler setup.
# Inputs: fig object you want the line graph attached to, acc_line data, gridspec value (position on fig, if multiple subplots present)
# Outputs: line ax object (line graph subplot)
def acc_plot_setup(fig, acc_line, gridspec=[1,1,1]):

    line_ax = fig.add_subplot(gridspec, polar=False)

    line_ax.plot(acc_line)

    return line_ax



# Plotting function for feature1 values. Feature 1 is classifier data that describes expected average raw EMG values for each channel for each gesture. This function plots rest, open, and close f1 values for a passed in participant. 
# Inputs: single armgame object for a subject (f1 data does not change between sessions, so only one is necessary to plot all relevant data for a subject), title of output figure, path to save figure to
def f1_radar_plot(ag_object, title, save_path):

    # Populate list of feature data to plot. Structure is features = [a,b,c] where a,b,c are one gesture's raw emg avgs for each of the 8 channels (a = [emg1_f1,emg2_f1,emg3_f1...emg8_f1])
    features = []
    features.append(ag_object.feature_df.loc[(0,1):(0,8),1].tolist())
    features.append(ag_object.feature_df.loc[(17,1):(17,8),1].tolist())
    features.append(ag_object.feature_df.loc[(18,1):(18,8),1].tolist())

    fig = plt.figure(figsize=(14,14)) # Create fig object

    # Generate category titles for radarplot, titles are made by appending the numbers 1-8 to the string emgChan
    categories = []
    for i in range(1,9):
        categories.append('emgChan' + str(i))
    
    polar_ax, ANGLES = radar_plot_setup(fig,categories) # set up radar plot using categories

    # Create labels for each gesture, will attach to individual lines of the plot
    labels = ['Rest','Open','Close']
    i = 0
    for feature in features: # Plots each feature's set of raw emg avgs on the radar plot
        feature += feature[:1] # append first element to back of list, convention is required in order to fully connect graph. Basically closes the plot, so you don't have empty space between the last and first element.
        polar_ax.plot(ANGLES,feature,linewidth=4, label = labels[i]) # Plot feature data on ANGLES, label with gesture
        polar_ax.scatter(ANGLES, feature, s=160) # Create scatterplot values to make vertices of graph more clear
        i += 1 # Increment gesture label iterator

    polar_ax.legend(loc='upper right') # place legend in desirable location

    fig.savefig(save_path) # save file to relevant path


def worm_accuracy_plot(ag_object, bounds, title, which_worm = 'raw_emg'):
    raw_emg_df = ag_object.armgame_df
    f1_diff_df = ag_object.f1_distances

    start = bounds[0]
    end = bounds[1]

    categories = []
    for i in range(1,9):
        categories.append('emgChan' + str(i))

    fig = plt.figure(figsize=(14,14))

    gs = fig.add_gridspec(4,3)

    polar_ax, ANGLES = radar_plot_setup(fig, categories,gs[0:2,:])

    acc_line_data = raw_emg_df.loc[start:end, 'rolling_acc_bin25']
    acc_line_data = acc_line_data.reset_index(drop=True)

    line_ax = acc_plot_setup(fig,acc_line_data,gs[3,:])

    def raw_emg_gif(idx):
        if len(polar_ax.lines) > 20:
            polar_ax.lines.remove(polar_ax.lines[1])

        line_ax.axvline(x=idx, ymin=.02, ymax=.98, color='r')

        if len(line_ax.lines) > 2:
            line_ax.lines.remove(line_ax.lines[1])

        values = raw_emg_df[categories].loc[start + idx].tolist()
        values += values[:1]
        polar_ax.plot(ANGLES, values, linewidth=4,label = categories)
        # ax.scatter(ANGLES, values, s=160)

        return [fig]

    def f1_diff_gif(idx):
        if len(polar_ax.lines) > 20:
            polar_ax.lines.remove(polar_ax.lines[0])

        line_ax.axvline(x=idx, ymin=.02, ymax=.98, color='r')

        if len(line_ax.lines) > 2:
            line_ax.lines.remove(line_ax.lines[1])

        values = f1_diff_df[range(1,9)].loc[start + idx].tolist()
        values += values[:1]
        polar_ax.plot(ANGLES,values,linewidth=4, label = categories)

        return [fig]

    frames = end - start 

    callback_func = raw_emg_gif
    if which_worm == 'f1_dist':
        print('f1_dist')
        #polar_ax.lines.remove(polar_ax.lines[0])
        callback_func = f1_diff_gif

    ani = FuncAnimation(fig,callback_func,frames=frames,interval=25,blit=True,repeat=True)

    file_path = 'worms/' + which_worm + '/' + title + '.gif'

    print('save ani: ' + file_path)

    ani.save(file_path,writer=PillowWriter(fps=10))

    print('done')





# Depreciated Function. Original draft of radar plotting function. Very specific use case, did not generalize well but functioned to output the graph I was looking for (and teach me more about plotting)
# Once I had this working, I tried designing a more flexible system to support different radar style and worm style outputs. This learning is reflected in the above functions, most notably worm_accuracy_plot 


# def radar_gif(armgame_object, feature1, bounds, title,which_worm):

#     raw_emg_df = armgame_object.df
#     f1_diff_df = armgame_object.f1_distances

#     print(bounds)
#     start = bounds[0]
    
#     end = bounds[1]

#     categories = []

#     for i in range(1,9):
#         categories.append('emgChan' + str(i))

#     # print(categories)

#     ANGLES = [n / len(categories) * 2 * np.pi for n in range(len(categories))]
#     ANGLES += ANGLES[:1]
#     feature1 += feature1[:1]

#     fig = plt.figure(figsize=(14,14))

#     fig.suptitle(title)

#     gs = fig.add_gridspec(4,3)

    
#     polar_ax = fig.add_subplot(gs[0:2,:], polar=True)
#     line_ax = fig.add_subplot(gs[3,:], polar=False)

#     #plt.subplots(2,1,gridspec_kw = {'height_ratios':[3,1]})

#     data = raw_emg_df.loc[start:end, 'rolling_acc_bin25']
#     data = data.reset_index(drop=True)

#     line_ax.plot(data)

#     polar_ax.set_theta_offset(np.pi / 2)
#     polar_ax.set_theta_direction(-1)


#     polar_ax.set_xticks(ANGLES[:-1])
#     polar_ax.set_xticklabels(categories, size=14)
    
#     polar_ax.plot(ANGLES,feature1,linewidth=4, label = categories, zorder=3, color = '#ffd11a')

#     def raw_emg_worm(idx):
#         if len(polar_ax.lines) > 25:
#             polar_ax.lines.remove(polar_ax.lines[1])

#         line_ax.axvline(x=idx, ymin=.02, ymax=.98, color='r')

#         if len(line_ax.lines) > 2:
#             line_ax.lines.remove(line_ax.lines[1])

#         values = raw_emg_df[categories].loc[start + idx].tolist()
#         values += values[:1]
#         polar_ax.plot(ANGLES, values, linewidth=4,label = categories)
#         # ax.scatter(ANGLES, values, s=160)

#         return [fig]

#     def f1_dist_worm(idx):
#         if len(polar_ax.lines) > 25:
#             polar_ax.lines.remove(polar_ax.lines[0])

#         line_ax.axvline(x=idx, ymin=.02, ymax=.98, color='r')

#         if len(line_ax.lines) > 2:
#             line_ax.lines.remove(line_ax.lines[1])

#         values = f1_diff_df[range(1,9)].loc[start + idx].tolist()
#         values += values[:1]
#         polar_ax.plot(ANGLES,values,linewidth=4, label = categories)

#         return [fig]

#     print('Begin Ani')

#     frames = end - start
#     print(frames)

#     callback_func = raw_emg_worm
#     if which_worm == 'f1_dist':
#         print('here')
#         polar_ax.lines.remove(polar_ax.lines[0])
#         callback_func = f1_dist_worm
        
#     ani = FuncAnimation(fig,callback_func,frames = frames, interval=25, blit=True, repeat=True)

#     file_path = 'worms/' + which_worm + '/' + title + '.gif'
#     print('save ani: ' + file_path)

#     ani.save(file_path,writer=PillowWriter(fps=10))

#     print('done')
