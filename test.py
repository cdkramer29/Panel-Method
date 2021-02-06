import fun
import matplotlib.pyplot as plt
import numpy as np
import datetime

start = datetime.datetime.now()

# Digit 1 - Maximum camber (% chord)
# Digit 2 - Distance of maximum camber from leading edge (tenths of the chord)
# Digits 3+4 - Max thickness of airfoil (% chord)

airfoil = "2415"
plot = True

# Convert 4 digit code to numbers
maxCamber = int(airfoil[0]) / 100
maxCamberLoc = int(airfoil[1]) / 10
maxThick = int(airfoil[2] + airfoil[3]) / 100

# Chord and angle of attack
chord = 0.25
span = 1
alpha = 10
v0 = 16

height = 0.5

rho = 1.225

# If inputs result in an error, try changing N_panels to fix my bad code
N_panels = 25
n = 100

# Make empty lists which we will fill up with stuff later

x = np.linspace(0, chord, n)

# Calculate Airfoil Coordinates

naca, camber = fun.airfoil_coord(chord, maxCamber, maxCamberLoc, maxThick, n, False)
nacarev, camberrev = fun.airfoil_coord(chord, maxCamber, maxCamberLoc, maxThick, n, True)


panels = fun.define_panels(nacarev[0], nacarev[1], N=N_panels)

# Calculate how long it took
end = datetime.datetime.now()
print("System took " + str((end - start).total_seconds() * 1000) + " ms")

if plot:
    # Plot airfoil shape and angle of attack
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(naca[0][0:int(1.5*n)], naca[1][0:int(1.5*n)], label='airfoil')
    ax1.plot(nacarev[0][0:int(1.5*n)], nacarev[1][0:int(1.5*n)], label='airfoil')
    ax1.set_aspect(1)  # Makes sure it doesn't look like it just crashed into a tree
    ax1.legend(loc='best', prop={'size': 16})

    ax2.grid()
    ax2.plot([panel.xc for panel in panels if panel.loc == 'upper'],
             [panel.yc for panel in panels if panel.loc == 'upper'],
             label='upper surface',
             color='r', linestyle='-', linewidth=2, marker='o', markersize=6)
    ax2.plot([panel.xc for panel in panels if panel.loc == 'lower'],
             [panel.yc for panel in panels if panel.loc == 'lower'],
             label='lower surface',
             color='b', linestyle='-', linewidth=1, marker='o', markersize=6)
    ax2.legend(loc='best', prop={'size': 16})
    ax2.set_aspect(1)

    plt.show()