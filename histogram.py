from sys import argv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOT_LIBRARY = 'plotly'

def seaborn_plot(data, subjects):
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(30, 30))
    for i, ax in enumerate(axs.flatten()):
        if i < len(subjects):
            sns.histplot(data, x=subjects[i], hue='Hogwarts House', bins=20, multiple='dodge', common_norm=False, stat='percent', ax=ax)
        else:
            ax.axis('off')
    fig.savefig('histogram.png')


def plotly_plot(data, subjects):
    fig = make_subplots(
    rows=4, cols=4, subplot_titles=subjects, shared_xaxes=False, shared_yaxes=False)
    trace_index = 0
    for i, subject in enumerate(subjects, start=0):
        subplot = px.histogram(data, x=data[subject], nbins=50, color='Hogwarts House', barmode='group', histnorm='percent')
        for trace in subplot.data: # for each subplot, 4 traces (one trace by house)
            fig.add_trace(trace, row=int(i/4)+1, col=int(i%4)+1)
            if i !=0:
                fig.data[trace_index].xaxis='x'+str(i+1)
                fig.data[trace_index].bingroup='x'+str(i+1)
                fig.data[trace_index].yaxis='y'+str(i+1)
                fig.data[trace_index].legendgroup = subject
                fig.data[trace_index].showlegend = (i == 0)
            trace_index += 1
    fig.update_layout(height=1000)
    fig.write_html("histogram.html")

def main():
    try:
        data = pd.read_csv("datasets/dataset_train.csv")
        subjects = ['Arithmancy', 'Astronomy', 'Herbology',
                    'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
                    'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
                    'Care of Magical Creatures', 'Charms', 'Flying']
        if PLOT_LIBRARY == 'seaborn':
            seaborn_plot(data, subjects)
        else:
            plotly_plot(data, subjects)
    except (AssertionError, Exception) as err:
        print("Error: ", err)

if __name__ == "__main__":
    main()
