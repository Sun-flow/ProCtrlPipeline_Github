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



# Make sure to pass in fig object you would like to manipulate
def radar_plot_setup(fig,categories,gridspec=[1,1,1]):
    spokes = len(categories)

    ANGLES = [n / spokes * 2 * np.pi for n in range(spokes)]
    ANGLES += ANGLES[:1]

    polar_ax = fig.add_subplot(gridspec, polar=True)

    polar_ax.set_theta_offset(np.pi / 2)
    polar_ax.set_theta_direction(-1)


    polar_ax.set_xticks(ANGLES[:-1])
    polar_ax.set_xticklabels(categories, size=14)

    return polar_ax, ANGLES

def acc_plot_setup(fig, acc_line, gridspec=[1,1,1]):

    line_ax = fig.add_subplot(gridspec, polar=False)

    line_ax.plot(acc_line)

    return line_ax


def f1_radar_plot(ag_object, title, save_path):
    features = []

    features.append(ag_object.feature_df.loc[(0,1):(0,8),1].tolist())
    features.append(ag_object.feature_df.loc[(17,1):(17,8),1].tolist())
    features.append(ag_object.feature_df.loc[(18,1):(18,8),1].tolist())

    fig = plt.figure(figsize=(14,14))

    categories = []
    for i in range(1,9):
        categories.append('emgChan' + str(i))
    polar_ax, ANGLES = radar_plot_setup(fig,categories)

    labels = ['Rest','Open','Close']
    i = 0
    for feature in features:        
        feature += feature[:1]
        polar_ax.plot(ANGLES,feature,linewidth=4, label = labels[i])
        polar_ax.scatter(ANGLES, feature, s=160)

        i += 1

    polar_ax.legend(loc='upper right')

    fig.savefig(save_path)


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






def radar_gif(armgame_object, feature1, bounds, title,which_worm):

    raw_emg_df = armgame_object.df
    f1_diff_df = armgame_object.f1_distances

    print(bounds)
    start = bounds[0]
    
    end = bounds[1]

    categories = []

    for i in range(1,9):
        categories.append('emgChan' + str(i))

    # print(categories)

    ANGLES = [n / len(categories) * 2 * np.pi for n in range(len(categories))]
    ANGLES += ANGLES[:1]
    feature1 += feature1[:1]

    fig = plt.figure(figsize=(14,14))

    fig.suptitle(title)

    gs = fig.add_gridspec(4,3)

    
    polar_ax = fig.add_subplot(gs[0:2,:], polar=True)
    line_ax = fig.add_subplot(gs[3,:], polar=False)

    #plt.subplots(2,1,gridspec_kw = {'height_ratios':[3,1]})

    data = raw_emg_df.loc[start:end, 'rolling_acc_bin25']
    data = data.reset_index(drop=True)

    line_ax.plot(data)

    polar_ax.set_theta_offset(np.pi / 2)
    polar_ax.set_theta_direction(-1)


    polar_ax.set_xticks(ANGLES[:-1])
    polar_ax.set_xticklabels(categories, size=14)
    
    polar_ax.plot(ANGLES,feature1,linewidth=4, label = categories, zorder=3, color = '#ffd11a')

    def raw_emg_worm(idx):
        if len(polar_ax.lines) > 25:
            polar_ax.lines.remove(polar_ax.lines[1])

        line_ax.axvline(x=idx, ymin=.02, ymax=.98, color='r')

        if len(line_ax.lines) > 2:
            line_ax.lines.remove(line_ax.lines[1])

        values = raw_emg_df[categories].loc[start + idx].tolist()
        values += values[:1]
        polar_ax.plot(ANGLES, values, linewidth=4,label = categories)
        # ax.scatter(ANGLES, values, s=160)

        return [fig]

    def f1_dist_worm(idx):
        if len(polar_ax.lines) > 25:
            polar_ax.lines.remove(polar_ax.lines[0])

        line_ax.axvline(x=idx, ymin=.02, ymax=.98, color='r')

        if len(line_ax.lines) > 2:
            line_ax.lines.remove(line_ax.lines[1])

        values = f1_diff_df[range(1,9)].loc[start + idx].tolist()
        values += values[:1]
        polar_ax.plot(ANGLES,values,linewidth=4, label = categories)

        return [fig]

    print('Begin Ani')

    frames = end - start
    print(frames)

    callback_func = raw_emg_worm
    if which_worm == 'f1_dist':
        print('here')
        polar_ax.lines.remove(polar_ax.lines[0])
        callback_func = f1_dist_worm
        
    ani = FuncAnimation(fig,callback_func,frames = frames, interval=25, blit=True, repeat=True)

    file_path = 'worms/' + which_worm + '/' + title + '.gif'
    print('save ani: ' + file_path)

    ani.save(file_path,writer=PillowWriter(fps=10))

    print('done')
