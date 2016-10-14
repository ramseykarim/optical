import matplotlib.pyplot as plt


def full_screen():
    for i in plt.get_fignums():
        plt.figure(i)
        fig_management = plt.get_current_fig_manager()
        fig_management.window.showMaximized()
        plt.tight_layout()


def show():
    full_screen()
    plt.show()
