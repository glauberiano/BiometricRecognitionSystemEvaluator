import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class Descritivas:
    @staticmethod
    def make_plot(dataset, var, save=False):
        df = dataset.copy()
        bacc_means, bacc_std = tuple(round(df['Bacc_mean'], 2)), tuple(round(df['Bacc_std'], 2))
        fmr_means, fmr_std = tuple(round(df['FMR_mean'], 2)), tuple(round(df['FMR_std'], 2))
        fnmr_means, fnmr_std = tuple(round(df['FNMR_mean'], 2)), tuple(round(df['FNMR_std'], 2))

        ind = np.arange(len(bacc_means))  # the x locations for the groups
        width = 0.25  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, bacc_means, width, yerr=bacc_std,
                        color='SkyBlue', label='Bacc')
        rects2 = ax.bar(ind + width, fmr_means, width, yerr=fmr_std,
                        color='IndianRed', label='fmr')
        rects3 = ax.bar(ind + 2*width, fnmr_means, width, yerr=fnmr_std,
                        color='yellow', label='fnmr')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title('Scores by ' + var)
        ax.set_xticks(ind+0.25)
        ax.set_xticklabels(tuple(df[var]))
        ax.legend()


        def autolabel(rects, xpos='center'):
            """
            Attach a text label above each bar in *rects*, displaying its height.

            *xpos* indicates which side to place the text w.r.t. the center of
            the bar. It can be one of the following {'center', 'right', 'left'}.
            """
        # import pdb; pdb.set_trace();
            xpos = xpos.lower()  # normalize the case of the parameter
            ha = {'center': 'center', 'right': 'left', 'left': 'right'}
            offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                        '{}'.format(height), ha=ha[xpos], va='bottom')


        autolabel(rects1, "left")
        autolabel(rects2, "right")
        autolabel(rects3, "center")

        if save == True:
            plt.savefig(os.path.join("graficos", var+"_mean.png"))
            plt.show()