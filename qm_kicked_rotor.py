"""(Quantum) kicked rotor.

Description
-----------

This animations visulizes the (quantum) kicked rotor in phase space. In the
quantum case the Husimi representation is used to form a comparable phase space
with the classical phase space.

Note
----

This python script uses external packages. In order to make the programm
work properly, one has to install the following packages:
- numpy
- matplotlib
- tqdm (if progressbar=True)
Additionally, if one want to use the "saveAni" feature, the program "ffmpeg"
have to be install on the computer and it should be callable in the console
via the command "ffmpeg".

Usage
-----

With a left click in the diagramm both the kicked rotor and quantum kicked
rotor will be started with the initial phase space configuration defined
by the position of the cursor. Have fun watching the animation :)
"""


import numpy as np
from matplotlib import pyplot as plt
import functools
import os
from scipy.stats import wasserstein_distance as wd


def standardmap(x0, p0, kicks=51, K=0.5):
    """Phase space distribution for a classical kicked rotor.

    Parameters
    ----------

    x0 : float
        Initial position.

    p0 : float
        Initial momentum.

    kicks : int (optional)
        Number of kicks. Default kicks=50

    K : float (optional)
        Kick strength. Default K=0.5

    Returns
    -------

    x : array
        Array of dimension "kicks" containing the x-coordinates after each kick
        (phase space).

    p : array
        Array of dimension "kicks" containing the p-coordinates after each kick
        (phase space).
    """

    x = np.ones(kicks)
    p = np.ones(kicks)

    x[0] = x0
    p[0] = p0

    for i in range(1, kicks):
        p[i] = p[i-1] + K*np.sin(x[i-1])
        x[i] = x[i-1] + p[i]

    p %= 2*np.pi
    x %= 2*np.pi

    return x, p


def mouse_click_event(event, ax, m, X_qm, Y_qm, kicks=50, K=0.5,
                        showClassic=False, saveAni=False, progressbar=False):
    """Method to start the animate of the phase space behavior of the
    (quantum) kicked rotor.

    Parameters
    ----------

    event : event
        "button_press_event"

    ax : matplotlib axes
        Specifies the axes in which the animation should take place.

    m : int
        Number of planck cells in one direction.

    X_qm : array
        x-coordinates of np.meshgrid

    Y_qm : array
        y-coordinates of np.meshgrid

    kicks : int (optional)
        Number of kicks. (Default 50)

    K : float (optional)
        Kick strength. (Default 0.5)

    showClassic : bool (optional)
        If "True" the animation will show the phase space distribution of the
        classical kicked rotor. If "False" these points will not be shown.
        (Default False)

    saveAni : bool (optional)
        If "True" every animation step will be saved as *.png in the directory
        of this python file (total of "kicks" images). After closing the
        plotting-window those images will be combined into a mp4 video using
        ffmpeg. (Default False)

    progressbar : bool (optional)
        If "True" the package "tqdm" will be imported. This results in a
        progressbar while the animation is calculated. (Default False)
    """

    mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes == ax and mode == "":
        xpos = event.xdata
        ypos = event.ydata

        try:
            ax.lines = []   ## TODO: make this thing work...
        except:
            pass

        if showClassic == True:
            X, P = standardmap(xpos, ypos, kicks, K)
            #ax.plot(X, P, marker=".", lw=0, color="r", zorder=1)
        elif showClassic == False:
            pass
        else:
            print("BOOLEAN ERROR: '{}' is not type bool."
            .format(str(showClassic)))

        x, p, psi_x, psi_p = wavepacket(m, xpos, ypos)
        V, T = phaseVectors(x, p, K)

        if progressbar == False:
            for kick in range(kicks):
                dist = husimi(m, psi_x)
                psi_x = timeEvolution(T, V, psi_x)
                ax.pcolor(X_qm, Y_qm, np.abs(dist)**2, zorder=-1)

                if showClassic == True:
                    classicPlot = ax.plot(X[kick], P[kick], marker=".", lw=0, color="r",
                    zorder=1)
                elif showClassic == False:
                    pass
                else:
                    print("BOOLEAN ERROR: '{}' is not type bool."
                    .format(str(showClassic)))

                ax.set_title(
                "Quantum Kicked Rotor (Kick = {0}; K = {1:1.1f})"
                .format(kick, K)
                )

                if saveAni == True:
                    plt.savefig("pic" + str(kick) + ".png")

                event.canvas.flush_events()
                event.canvas.draw()

        elif progressbar == True:
            import tqdm
            for kick in tqdm.tqdm(range(kicks)):
                dist = husimi(m, psi_x)
                psi_x = timeEvolution(T, V, psi_x)
                ax.pcolor(X_qm, Y_qm, np.abs(dist)**2, zorder=-1)

                if showClassic == True:
                    classicPlot = ax.plot(X[kick], P[kick], marker=".", lw=0, color="r",
                    zorder=1)
                elif showClassic == False:
                    pass
                else:
                    print("BOOLEAN ERROR: '{}' is not type bool."
                    .format(str(showClassic)))

                ax.set_title(
                "Quantum Kicked Rotor (Kick = {0}; K = {1:1.1f})"
                .format(kick, K)
                )

                if saveAni == True:
                    plt.savefig("pic" + str(kick) + ".png")

                event.canvas.flush_events()
                event.canvas.draw()

        else:
            print("BOOLEAN ERROR: '{}' is not type bool."
            .format(str(progressbar)))


def wavepacket(m, x0, p0):
    """Generates normalized Gaussian wavepacket in position and momentum space.

    Parameters
    ----------

    m : int
        Number of planck cells in one direction.

    x0 : float
        Initial x-coordinate of localized wavepacket in position space.


    p0 : float
        Initial p-coordinate of localized wavepacket in momentum space.

    Returns
    -------

    x : array
        Array of discret x-coordinates corresponding to the Planck cells.

    p : array
        Array of discret p-coordinates corresponding to the Planck cells.

    psi_x : array
        Array of wavepacket in position space.

    psi_p : array
        Array of wavepacket in momentum space.
    """

    hbar = 2*np.pi/m
    n = np.arange(m)
    x = 2*np.pi*n/m
    p = 2*np.pi*n/m

    psi_x = np.ones(m).astype(complex)
    psi_p = np.ones(m).astype(complex)

    psi_x[:] = np.exp(-(x-x0)**2/(2*hbar) + 1j*x*p0/hbar)
    norm_x = np.linalg.norm(psi_x)
    psi_x /= norm_x

    psi_p = np.fft.fft(psi_x, norm="ortho")

    return x, p, psi_x, psi_p


def husimi(m, psi_x):
    """Calculates the Husimi representation of the quantum phase space.

    Credits: https://github.com/mcooper12590/QuantumMaps

    Parameters
    ----------

    m : int
        Number of planck cells in one direction.

    psi_x : array
        Array of wavepacket in position space.

    Returns
    -------

    dist : array
        (m x m)-array of the Husimi phase space distribution.
    """

    dist = np.ones((m, m)).astype(complex)

    for i in np.arange(m):
        x0 = 2*np.pi*i/m
        x, p, psi_n, psi_p = wavepacket(m, x0, 0)
        dist[:, i] = np.fft.fft(np.conj(psi_n)*psi_x)

    return dist


def phaseVectors(x, p, K):
    """Calculates two vectors containing the values of the time evolution
    operator splitted into two factors, one factor describes the kinetic and
    the other describes the potential term.

    Parameters
    ----------

    x : array
        Array of dimension "kicks" containing the x-coordinates after each kick
        (phase space).

    p : array
        Array of dimension "kicks" containing the p-coordinates after each kick
        (phase space).

    K : float
        Kick strength.

    Returns
    -------

    V : array
        Potential term.

    T : array
        Kinetic term.
    """

    hbar = 2*np.pi/x.size
    V = np.exp(-1j/hbar*K*np.cos(x))
    T = np.exp(-1j/2/hbar*p*p)

    return V, T


def timeEvolution(T, V, psi_x):
    """Performs a one-time-step time evolution.

    Parameters
    ----------

    T : array
        Kinetic term. See "phaseVectors" for more details.

    V : array
        Potential term. See "phaseVectors" for more details.

    psi_x : array
        Array of wavepacket in position space.

    Returns
    -------

    inverseFT : array
        New eigenstate after the time step.
    """

    FT = np.fft.fft(V*psi_x)
    inverseFT = np.fft.ifft(T*FT)
    inverseFT /= psi_x.size

    return inverseFT


def main():
    """Initiate all functions and set up the animation.

    Parameters
    ----------

    K : float
        Kick strength.

    m : int
        Number of planck cells in one direction.

    kicks : int
        Number of kicks.

    x0 : float
        Initial position of the classical kicked rotor. Together with p0 and a
        kick strength of K=4.7 this initial condition lead to a integrable
        island in the phase space.

    p0 : float
        Initial momentum of the classical kicked rotor. Together with x0 and a
        kick strength of K=4.7 this initial condition lead to a integrable
        island in the phase space.

    saveAni : bool
        If "True" every animation step will be saved as *.png in the directory
        of this python file (total of "kicks" images). After closing the
        plotting-window those images will be combined into a mp4 video using
        ffmpeg.

    showClassic : bool
        If "True" the animation will show the phase space distribution of the
        classical kicked rotor. If "False" these points will not be shown.

    progressbar : bool
        If "True" the package "tqdm" will be imported. This results in a
        progressbar while the animation is calculated.

    Note
    ----

    This python script uses external packages. In order to make the programm
    work properly, one has to install the following packages:
    - numpy
    - matplotlib
    - tqdm (if progressbar=True)
    Additionally, if one want to use the "saveAni" feature the program "ffmpeg"
    have to be install on the computer and it should be callable in the console
    via the command "ffmpeg".

    Usage
    -----

    With a left click in the diagramm both the kicked rotor and quantum kicked
    rotor will be started with the initial phase space configuration defined
    by the position of the cursor. Have fun watching the animation :)
    """

    # -- Changeable parameters --
    K = 2.5
    m = 100
    kicks = 11
    x0 = 2.31                                       # coordinates of integrable
    p0 = 4.47                                       # island for K=4.7
    saveAni = True
    showClassic = False
    progressbar = False
    # -- Changeable parameters -- END --

    parameters = ["K", "m", "kicks", "x0", "p0", "saveAni", "showClassic",
                "progressbar"]
    values = [K, m, kicks, x0, p0, saveAni, showClassic, progressbar]

    print(__doc__)
    print("Current parameters")
    print("------------------\n")
    for parameter, value in zip(parameters, values):
        print(parameter + " = " + str(value))
    print("------------------\n")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, aspect=1.0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$p$")
    ax.set_title("Quantum Kicked Rotor (Kick = 0; K = {:1.1f})".format(K))
    plt.xlim([0, 2*np.pi*(1-1/m)])
    plt.ylim([0, 2*np.pi*(1-1/m)])

    # integrable region for K=4.7
    if K == 4.7:
        THETA, P = standardmap(x0, p0, kicks, K)
        ax.plot(THETA, P, marker="s", color="b", lw=0, markerfacecolor="none",
                markeredgecolor="b")

    # meshgrid
    X_qm, Y_qm = np.meshgrid(2*np.pi*np.arange(m)/m, 2*np.pi*np.arange(m)/m)

    # connection to the mouse click event
    on_click = functools.partial(mouse_click_event, ax=ax, K=K, m=m,
                                kicks=kicks, X_qm=X_qm, Y_qm=Y_qm,
                                showClassic=showClassic, saveAni=saveAni,
                                progressbar=progressbar)


    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()

    if saveAni == True:
        os.system(
            "ffmpeg -r 5 -f image2 -s 1920x1080 -i pic%d.png -vcodec libx264\
            -crf 25  -pix_fmt yuv420p quantum_kicked_rotor.mp4"
        )
    elif saveAni == False:
        pass
    else:
        print("BOOLEAN ERROR: '{}' is not type bool.".format(str(saveAni)))

if __name__ == "__main__":
    main()
