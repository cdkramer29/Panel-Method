"""

Airfoil Optimizer V2

So far it doesnt optimize anything
To do that we need to calculate cd so we can compare parameters to this

Program is actually fairly accurate for calculating cl, but only in the linear region.
It does not account for flow separation (as of now)
This means that if we tried to optimize now, we would just end up with a vertical brick


"""


import fun
import matplotlib.pyplot as plt
import numpy as np
import datetime

start = datetime.datetime.now()

# Digit 1 - Maximum camber (% chord)
# Digit 2 - Distance of maximum camber from leading edge (tenths of the chord)
# Digits 3+4 - Max thickness of airfoil (% chord)

airfoil = "2415"
plot = False

# Convert 4 digit code to numbers
maxCamber = int(airfoil[0]) / 100
maxCamberLoc = int(airfoil[1]) / 10
maxThick = int(airfoil[2] + airfoil[3]) / 100

# Chord and angle of attack
chord = 1
alpha = 10
v0 = 0.75

# If inputs result in an error, try changing N_panels to fix my bad code
N_panels = 50
n = 1000

# Make empty lists which we will fill up with stuff later

x = np.linspace(0, chord, n)

a = []
cList = []
# Calculate Airfoil Coordinates
for alpha in range(-10,10):
    naca, camber = fun.airfoil_coord(chord, maxCamber, maxCamberLoc, maxThick, n)

    # Rotate Coordinates
    # naca = rotate(naca,alpha)
    # camber = rotate(camber,alpha)

    panels = fun.define_panels(naca[0], naca[1], N=N_panels)

    # define freestream conditions
    freestream = fun.Freestream(u_inf=v0, alpha=alpha)

    A_source = fun.source_contribution_normal(panels)
    B_vortex = fun.vortex_contribution_normal(panels)

    A = fun.build_singularity_matrix(A_source, B_vortex)
    b = fun.build_freestream_rhs(panels, freestream)

    # solve for singularity strengths
    strengths = np.linalg.solve(A, b)

    # store source strength on each panel
    for i, panel in enumerate(panels):
        panel.sigma = strengths[i]

    # store circulation density
    gamma = strengths[-1]

    # tangential velocity at each panel center.
    fun.compute_tangential_velocity(panels, freestream, gamma, A_source, B_vortex)

    # surface pressure coefficient
    fun.compute_pressure_coefficient(panels, freestream)

    # calculate the accuracy (the closer to zero the better) by summing vortex/source strengths
    # (should sum to zero, or else mass is absorbed or released from airfoil)
    accuracy = sum([panel.sigma * panel.length for panel in panels])
    print('Sum of singularity strengths: {:0.6f}'.format(accuracy))

    # compute the chord and lift coefficient
    c = abs(max(panel.xa for panel in panels) -
            min(panel.xa for panel in panels))
    cl = (gamma * sum(panel.length for panel in panels) /
          (0.5 * freestream.u_inf * c))
    print('Lift coefficient: CL = {:0.3f}'.format(cl))
    print(alpha)
    a.append(alpha)
    cList.append(cl)
# Print out stuff
# print("Max camber: " + str(maxCamber) + " chord")
# print("Max camber location: " + str(maxCamberLoc) + " chord")
# print("Max airfoil thickness: " + str(maxThick) + " chord")

plt.plot(a,cList)
plt.show()

# Rotate Coordinates for viewing (rotated for calculation eariler)
nacaRot = fun.rotate(naca, alpha)
camberRot = fun.rotate(camber, alpha)

# Calculate how long it tool
end = datetime.datetime.now()
print("System took " + str((end - start).total_seconds() * 1000) + " ms")

if plot:
    # Plot airfoil shape and angle of attack
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(nacaRot[0], nacaRot[1])
    ax1.plot(camberRot[0], camberRot[1])
    ax1.plot([0, camberRot[0][-1]], [0, camberRot[1][-1]])
    ax1.set_aspect(1)  # Makes sure it doesn't look like it just crashed into a tree

    # Plot Pressures
    ax2.grid()
    ax2.plot([panel.xc for panel in panels if panel.loc == 'upper'],
             [panel.cp for panel in panels if panel.loc == 'upper'],
             label='upper surface',
             color='r', linestyle='-', linewidth=2, marker='o', markersize=6)
    ax2.plot([panel.xc for panel in panels if panel.loc == 'lower'],
             [panel.cp for panel in panels if panel.loc == 'lower'],
             label='lower surface',
             color='b', linestyle='-', linewidth=1, marker='o', markersize=6)
    ax2.legend(loc='best', prop={'size': 16})

    # Plot panel endpoints (too see if panels were correctly generated or if it drove off a cliff)
    ax3.plot(naca[0], naca[1])
    ax3.scatter([p.xa for p in panels], [p.ya for p in panels])
    ax3.set_aspect(1)

    plt.show()