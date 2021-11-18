import matplotlib.pyplot as plt

def get_plot(data,title,xlabel='Epochs',ylabel='Loss',savefig=True,showfig=False):
    plt.plot(data,'-bo')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(title+'.png')
