# coding=utf-8
def swap_axes(ax):
    import matplotlib.pyplot as plt

    lines = ax.get_lines()

    for line in lines:
        x_data_old = line.get_xdata()
        y_data_old = line.get_ydata()
        line.set_xdata(y_data_old)
        line.set_ydata(x_data_old)

    xlabel_old = ax.get_xlabel()
    ylabel_old = ax.get_ylabel()
    ax.set_xlabel(ylabel_old)
    ax.set_ylabel(xlabel_old)

    xlim_old = ax.get_xlim()
    ylim_old = ax.get_ylim()
    ax.set_xlim(ylim_old)
    ax.set_ylim(xlim_old)

    plt.draw()
    pass
