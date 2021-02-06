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

airfoil = "2412"
plot = False

# Convert 4 digit code to numbers
maxCamber = int(airfoil[0]) / 100
maxCamberLoc = int(airfoil[1]) / 10
maxThick = int(airfoil[2] + airfoil[3]) / 100

# Chord and angle of attack
chord = 0.25
span = 1
alpha = -15
v0 = 16

min_height = 0.05
max_height = 0.3

rho = 1.225

# If inputs result in an error, try changing N_panels to fix my bad code
N_panels = 50   # Going over 50 breaks things
n = 1000

# Make empty lists which we will fill up with stuff later

cl_list = []
height_list = []

heights = np.linspace(min_height, max_height, num=10)

for h in heights:

    x = np.linspace(0, chord, n)

    # Calculate Airfoil Coordinates

    naca, camber = fun.airfoil_coord(chord, maxCamber, maxCamberLoc, maxThick, n, True)

    # Rotate Coordinates
    naca = fun.rotate(naca, alpha)
    camber = fun.rotate(camber, alpha)
    naca = [naca[0], [naca[1][i]+h for i in range(0, len(naca[1]))]]

    panels = fun.define_panels(naca[0], naca[1], N=N_panels)

    naca2, camber2 = fun.airfoil_coord(chord, maxCamber, maxCamberLoc, maxThick, n, False)

    naca2 = fun.rotate(naca2, -alpha)
    camber2 = fun.rotate(camber2, -alpha)
    naca2 = [naca2[0], [naca2[1][i]-h for i in range(0, len(naca2[1]))]]

    panels2 = fun.define_panels(naca2[0], naca2[1], N=N_panels)

    # define freestream conditions
    freestream = fun.Freestream(u_inf=v0, alpha=0)

    total_panels = panels

    for i in range(0, len(panels)):
        total_panels = np.append(total_panels, panels2[i])

    A_source = fun.source_contribution_normal(total_panels)
    B_vortex = fun.vortex_contribution_normal(total_panels)

    A = fun.build_singularity_matrix(A_source, B_vortex)
    b = fun.build_freestream_rhs(total_panels, freestream)

    # solve for singularity strengths
    strengths = np.linalg.solve(A, b)

    # store source strength on each panel
    for i, panel in enumerate(total_panels):
        panel.sigma = strengths[i]

    # store circulation density
    gamma = strengths[-1]

    # tangential velocity at each panel center.
    fun.compute_tangential_velocity(total_panels, freestream, gamma, A_source, B_vortex)

    # surface pressure coefficient
    fun.compute_pressure_coefficient(total_panels, freestream)

    # calculate the accuracy (the closer to zero the better) by summing vortex/source strengths
    # (should sum to zero, or else mass is absorbed or released from airfoil)
    accuracy = sum([panel.sigma * panel.length for panel in total_panels])
    print('Sum of singularity strengths: {:0.6f}'.format(accuracy))

    # compute the chord and lift coefficient
    c = abs(max(panel.xa for panel in panels) -
            min(panel.xa for panel in panels))
    lift = span * sum((abs(panel.xa - panel.xb) * panel.cp * 0.5 * rho * v0**2) for panel in panels)
    cl = lift / (chord * span * 0.5 * rho * v0**2)

    cl_list.append(cl)
    height_list.append(h/c)


    print('Lift coefficient: CL = {:0.3f}'.format(cl))
    print('Lift = {:0.3f}'.format(lift))
    print('H/C = {:0.3f}'.format(h/c))

plt.plot(height_list, cl_list)
plt.show()

# Print out stuff
# print("Max camber: " + str(maxCamber) + " chord")
# print("Max camber location: " + str(maxCamberLoc) + " chord")
# print("Max airfoil thickness: " + str(maxThick) + " chord")


# Calculate how long it tool
end = datetime.datetime.now()
print("System took " + str((end - start).total_seconds() * 1000) + " ms")

if plot:
    # Plot airfoil shape and angle of attack
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(naca[0], naca[1], label='airfoil')
    ax1.plot(naca2[0], naca2[1], label='mirror')
    ax1.set_aspect(1)  # Makes sure it doesn't look like it just crashed into a tree
    ax1.legend(loc='best', prop={'size': 16})

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

    ax3.grid()
    ax3.plot([panel.xc for panel in total_panels if panel.loc == 'upper'],
             [panel.yc for panel in total_panels if panel.loc == 'upper'],
             label='upper surface',
             color='r', linestyle='-', linewidth=2, marker='o', markersize=6)
    ax3.plot([panel.xc for panel in total_panels if panel.loc == 'lower'],
             [panel.yc for panel in total_panels if panel.loc == 'lower'],
             label='lower surface',
             color='b', linestyle='-', linewidth=1, marker='o', markersize=6)
    ax3.legend(loc='best', prop={'size': 16})
    ax3.set_aspect(1)

    plt.show()