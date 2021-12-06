# Quantum-Kicked-Rotor
Python tool which visualizes the quantum kicked rotor in phase space, meaning
in the Husimi representation.
## Examples
For those two examples I have chosen K=2.5, m=100 and kicks=51.
### Example: Chaotic motion
![Example animation](https://github.com/Bra-A-Ket/Quantum-Kicked-Rotor/blob/main/chaoticmotion.gif)
### Example: Integrable motion
![Example animation](https://github.com/Bra-A-Ket/Quantum-Kicked-Rotor/blob/main/integrablemotion.gif)
## Features
- Starts animation for the time evolution of the quantum kicked rotor via mouse
klick in the plot. The mouse position will be chosen as the initial center
of the initial wave packet (maximally located Gaussian)
- If requested, compares the quantum time evolution to the trajectory of the
classical kicked rotor.
- If requested, saves every image in the time evolution as *.png and creates a
*.mp4 out of it.
## Usage
### Required packages
This program works with external packages, which are the following:
- numpy
```bash
python3 -m pip install numpy
```
- matplotlib
```bash
python3 -m pip install matplotlib
```
- tqdm (optional, only if progressbar=True)
```bash
python3 -m pip install tqdm
```
### FFmpeg
FFmpeg is only needed, if you want to save the animation.
- For Windows simply visit the official ffmpeg website, download and install it.
Make sure that ffmpeg is callable in the console via
```bash
ffmpeg
```
- For Ubuntu
```bash
sudo apt-get install ffmpeg
```
## Changeable variables
- K : float - Kick strength.

- m : int - Number of planck cells in one direction.

- kicks : int - Number of kicks.

- x0 : float - Initial position of the classical kicked rotor. Together with p0
and a kick strength of K=4.7 this initial condition lead to a integrable
island in the phase space.

-  p0 : float - Initial momentum of the classical kicked rotor. Together with x0
and a kick strength of K=4.7 this initial condition lead to a integrable
island in the phase space.

-  saveAni : bool - If "True" every animation step will be saved as *.png in the
 directory of this python file (total of "kicks" images). After closing the
plotting-window those images will be combined into a mp4 video using ffmpeg.

-  showClassic : bool - If "True" the animation will show the phase space
distribution of the classical kicked rotor. If "False" these points will not be shown.

-  progressbar : bool - If "True" the package "tqdm" will be imported. This
results in a progressbar while the animation is calculated.
- Default values:
```python
K = 2.5
m = 100
kicks = 11
x0 = 2.31                                       # coordinates of integrable
p0 = 4.47                                       # island for K=4.7
saveAni = True
showClassic = False
progressbar = False
```
