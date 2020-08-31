import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

def plot_scores(scores, n_games):
    x = np.arange(start=1, stop=n_games+1)
    scores = np.array(scores)
    # make plot
    line_plot = sb.lineplot(x, scores)
    # line_plot.set(x = 'n_games', y = 'score')
    # line_plot.set_title('agent score vs n_games')
    plt.show()