from math import *
import numpy as np
import random
import Tkinter as tk
import ttk as ttk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os, sys, subprocess
import shelve

# pyinstaller -w --onefile --windowed TeamBuilder.py

global team_edges, num_interact, store_teams, ppl, teams, rnds, TeamsVal, PplVal, RndsVal, rank

##################################
# function that choose the teams #
##################################

def chooseteams():
    global team_edges, num_interact, store_teams, ppl, teams, rnds, TeamsVal, PplVal, RndsVal, ppl_teams

    maxloop = 10 ** 4

    # get the numbers entered
    ppl = int(PplVal.get())
    teams = int(TeamsVal.get())
    rnds=int(RndsVal.get())

    # calculate the number of teams and members in each
    mem_big = int(ceil(float(ppl) / teams))
    teams_big = ppl - (mem_big - 1) * teams
    mem_small = mem_big - 1
    teams_small = teams - teams_big

    # create array to store teams and number or interactions
    store_teams = np.zeros((rnds, ppl))
    num_interact = np.zeros((ppl, ppl))
    ppl_teams=np.zeros((ppl,rnds))

    team_edges = np.zeros((teams, 2)) # store where each team starts/stops
    mat = np.zeros((ppl, ppl)) # used to calcaulte how good a particular team choice is

    for iterbig in range(teams_big):
        index = [iterbig*mem_big, (iterbig+1)*mem_big-1]
        team_edges[iterbig, 0] = index[0]
        team_edges[iterbig, 1] = index[1]
        mat[index[0]:index[1]+1, index[0]:index[1]+1] = 1

    for itersmall in range(teams_small):
        index = [teams_big*mem_big+itersmall*mem_small, teams_big*mem_big+(itersmall+1)*mem_small-1]
        team_edges[itersmall+teams_big, 0] = index[0]
        team_edges[itersmall+teams_big, 1] = index[1]
        mat[index[0]:index[1]+1, index[0]:index[1]+1] = 1

    team_edges = team_edges.astype(int)
    mat = mat-np.identity(ppl)


    def my_shuffle(array):
        random.shuffle(array)
        return array


    # Calculate each round
    max_min_ints = np.zeros((rnds, 2))
    fit = np.zeros((rnds, 2))

    # Form the optimal teams
    for rnd_num in range(rnds):
        # create vector of people and randomise, so no grouping is favoured
        arrange = range(ppl)
        arrange = my_shuffle(arrange)

        if rnd_num == 0:  # form the first team
            store_teams[0, :] = arrange

        else:  # form the remaining teams
            # Rearrange num_interact according to random order
            add_to = num_interact[:, arrange]
            add_to = add_to[arrange, :]

            # Calcualte the best case we could reach in the team choice
            num_vals = sum(sum(mat)).astype(int)
            a= np.sort(num_interact.flatten())
            perfect = sum(np.square(a[0:num_vals]))

            # add up all the interactions we would currently add to. Square them to discourage adding to large interactions
            optim1 = sum(sum(np.multiply(np.square(num_interact), mat)))

            # loop through possible alternative teams
            loopiter = 1
            while loopiter < maxloop:
                swap = np.random.randint(ppl, size=2)
                swap = swap[0], swap[1]
                fliped = swap[1], swap[0]
                add_to2 = add_to
                add_to2[swap, :] = add_to2[fliped, :]
                add_to2[:, swap] = add_to2[:, fliped]
                optim2 = sum(sum(np.multiply(np.square(add_to2), mat)))

                if optim2 < optim1:  # If this is a better team than the last loop, keep it
                    add_to = add_to2
                    holdval = arrange[swap[0]]
                    arrange[swap[0]] = arrange[swap[1]]
                    arrange[swap[1]] = holdval
                    optim1 = optim2

                if optim2 == perfect:  # If it can't get any better, exit loop
                    loopiter = maxloop

                loopiter = loopiter + 1

            # store the desired perfect value alongside the true optim1 value
            fit[rnd_num, 0] = perfect
            fit[rnd_num, 1] = optim1

            # Store the new team
            store_teams[rnd_num, :] = arrange

        # store the new interactions
        for interact_iter in range(teams):
            index = arrange[int(team_edges[interact_iter, 0]):(int(team_edges[interact_iter, 1])+1)]
            ppl_teams[index, rnd_num] = interact_iter + 1
            for indexiter in index:
                num_interact[index, indexiter] = num_interact[index, indexiter] + 1

        max_min_ints[rnd_num, 0] = num_interact.min()
        mask = np.ones(num_interact.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        max_min_ints[rnd_num, 1] = num_interact[mask].max()

    savesession('ppl_teams',ppl_teams)
    savesession('team_edges', team_edges)
    savesession('store_teams', store_teams)
    savesession('ppl', ppl)
    savesession('teams', teams)
    savesession('rnds', rnds)
    savesession('team_scores', np.zeros((rnds, teams)))
    create_pdf(ppl_teams, ppl, teams, rnds)
    scorewindow(ppl, rnds, teams)

##################################################
# function that calculates the individual scores #
##################################################

def teamscores():
    global sorted_ppl, sorted_scores, individual_scores, scoresframe, teams, rnds, tree, place

    # store all the scores
    team_scores = np.zeros((rnds, teams))
    for m in range(rnds):
        for n in range(teams):
            A = scoresframe.entries[m*teams+n].get()
            if len(A) >= 1:
                team_scores[m,n] = float(A)

    # calculate the individual scroes
    individual_scores = np.zeros((ppl, 1))

    for team_iter in range(teams):
        for rnd_iter in range(rnds):
            members = store_teams[rnd_iter, team_edges[team_iter, 0]:(team_edges[team_iter, 1]+1)].astype(int)
            individual_scores[members] = individual_scores[members] + team_scores[rnd_iter, team_iter]

    sorted_ppl = np.argsort(individual_scores, 0)[::-1]
    sorted_scores = np.sort(individual_scores, 0)[::-1]
    place=np.argsort(sorted_ppl, 0)

    displayscores(sorted_ppl, sorted_scores, range(ppl), tree, ppl)
    savesession('team_scores', team_scores)
    raise_frame(results)

###################################
# Display the team choices in PDF #
###################################

def create_pdf(ppl_teams, ppl, teams, rnds):
    perpage=4
    pp = PdfPages('teamallocation.pdf')
    piter=0
    fig, axs = plt.subplots(perpage, 1)
    for miter in range(perpage):
        axs[miter].axis('tight')
        axs[miter].axis('off')

    for person in range(ppl):
        clust_data = np.zeros([2,rnds], dtype=int)
        for iter in range(rnds):
            clust_data[0,iter]=iter+1
            clust_data[1,iter]=ppl_teams[person,iter]
        rowlabel = ("round","team")
        axs[piter].table(cellText=clust_data, rowLabels=rowlabel, loc='center')
        axs[piter].set_title("Participant %2d" %(person+1), y=0.7)
        piter=piter+1

        if round(float(person+1)/perpage)==float(person+1)/perpage or person+1==ppl:
            piter=0
            pp.savefig()
            plt.cla
            #fig, axs = plt.subplots(perpage, 1)
            for miter in range(perpage):
                axs[miter].axis('tight')
                axs[miter].axis('off')
            
    pp.close()
    openpdf()
    #plt.close()
    

def openpdf():
    if sys.platform == "win32":
        os.startfile('teamallocation.pdf')
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, 'teamallocation.pdf'])

###########################################
# Function that saves the current session #
###########################################

def savesession(name, variable):
    my_shelf = shelve.open('TeamBuilder_save')
    my_shelf[name]=variable
    my_shelf.close()

def loadsession():
    global team_edges, ppl_teams, store_teams, ppl, teams, rnds, team_scores

    my_shelf = shelve.open('TeamBuilder_save')
    for key in my_shelf:
        globals()[key] = my_shelf[key]
    my_shelf.close()

    scorewindow(ppl, rnds, teams)

    #add previous scores
    for m in range(rnds):
         for n in range(teams):
             scoresframe.entries[m * teams + n].insert(tk.END, str(team_scores[m,n]))

    openpdf()
    teamscores()

#########################################
# Some functions needed for the display #
#########################################

def raise_frame(frame):
    frame.tkraise()

def displayscores(sorted_ppl, individualscores, place, tree, ppl):
    tree.delete(*tree.get_children())
    for iter in range(ppl):
        rowvals=(int(place[iter]+1),int(sorted_ppl[iter]+1),float(individualscores[iter]))
        tree.insert("", "end", text="", values=rowvals)

###############################$
# create the different Windows #
################################

def loadwindow():

    infoFrame = tk.Frame(loadF)
    infoFrame.pack()
    #someinfo = tk.Label(infoFrame, text='Team Builder is a program which chooses the teams for a rotation style quiz night, where the teams change every round. After the teams are selected', width=10).pack()

    loadFrame = tk.Frame(loadF)
    loadFrame.pack()
    subFrame=tk.Frame(loadFrame)
    subFrame.pack()
    tk.Button(subFrame, text="Load Last Session", command=loadsession).grid(row=0, column=0)
    tk.Button(subFrame, text="Start New Session", command=infowindow).grid(row=1, column=0)
    tk.Label(subFrame, text='Warning: starting a new session will overwrite your last session').grid(row=2, column=0)
    raise_frame(loadF)

def infowindow():
    global TeamsVal, PplVal, RndsVal
    # create some frames
    InfoFrame = tk.Frame(window)
    InfoFrame.pack()
    RunFrame = tk.Frame(window)
    RunFrame.pack(side=tk.BOTTOM)

    # create boxes to enter variables

    PplDisp = tk.Label(InfoFrame, text="Number of People=").grid(row=0, column=0)
    PplVal = tk.Entry(InfoFrame, width=5)
    PplVal.grid(row=0, column=1)

    TeamsDisp = tk.Label(InfoFrame, text="Number of Teams=").grid(row=1, column=0)
    TeamsVal = tk.Entry(InfoFrame, width=5)
    TeamsVal.grid(row=1, column=1)

    RndsDisp = tk.Label(InfoFrame, text="Number of Rounds=").grid(row=2, column=0)
    RndsVal = tk.Entry(InfoFrame, width=5)
    RndsVal.grid(row=2, column=1)
    #RndsMax=tk.Label(InfoFrame, text='(max=30)').grid(row=2, column=2)

    # create button to calculate team
    tk.Button(RunFrame, text="Calculate Teams", command=chooseteams).pack()
    tk.Label(RunFrame, text='Note: It may take a few minutes to calculate the optimal teams.').pack()
    raise_frame(window)

def scorewindow(ppl, rnds, teams):
    global scoresframe, tree
    resultswindow(ppl)
    scoresframe = tk.Frame(scores)
    scoresframe.pack(side=tk.TOP)
    calcframe = tk.Frame(scores)
    calcframe.pack(side=tk.BOTTOM)

    tk.Button(calcframe, text="Calculate Scores (& save)", command=teamscores).pack()

    #make entry boxes for each team score
    scoresframe.entries = []
    for n in range(teams):
        # create left side info labels
        tk.Label(scoresframe, text="Team %2d: " % int(n+1)).grid(row=n+1, column=0)

    if rnds<=25:
        w=4
    else:
        w=3

    for m in range(rnds):
         for n in range(teams):
            # create entries list
            scoresframe.entries.append(tk.Entry(scoresframe, width=w))
            # grid layout the entries
            scoresframe.entries[m*teams+n].grid(row=n+1, column=m+1)


    for m in range(rnds):
        tk.Label(scoresframe, text="%2d" % int(m+1)).grid(row=0, column=m+1)
    tk.Label(scoresframe, text="Round:").grid(row=0, column=0)

def resultswindow(ppl):
    global tree, results, place, individual_scores, sorted_ppl, sorted_scores

    rootresults = tk.Tk()
    rootresults.title("Results")
    results = tk.Frame(rootresults)
    results.grid(row=0, column=0, sticky='news')

    scrollbar = ttk.Scrollbar(results, orient="vertical")
    scrollbar.pack(side="right", fill="y")

    tree = ttk.Treeview(results, height=np.minimum(ppl,30), yscrollcommand=scrollbar.set)
    scrollbar.config(command=tree.yview)
    tree.pack(anchor='center')

    tree["columns"] = ("first", "second", "third")
    tree.column("first", width=70)
    tree.column("second", width=70)
    tree.column("third", width=70)

    tree.heading('first', text='Place', command=lambda: \
                 displayscores(sorted_ppl, sorted_scores, range(ppl), tree, ppl))
    tree.heading('second', text='Person', command=lambda: \
                 displayscores(range(ppl), individual_scores, place, tree, ppl))
    tree.heading('third', text='Score', command=lambda: \
                 displayscores(sorted_ppl, sorted_scores, range(ppl), tree, ppl))

    tree['show'] = 'headings'
    raise_frame(scores)

###################
# Run the program #
###################

root=tk.Tk()
root.title("TeamBuilder")

# define the windows
loadF = tk.Frame(root)
window = tk.Frame(root)
scores = tk.Frame(root)

for frame in (loadF, window, scores):
    frame.grid(row=0, column=0, sticky='news')

loadwindow()
root.mainloop() # Loops the window to prevent the window from just "flashing once"