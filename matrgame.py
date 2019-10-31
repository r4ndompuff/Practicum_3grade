import numpy as np

#FUNCTIONS

#def nash_equilibrium(a):

def is_saddle(mtr):
    global min
    global max
    rows, columns = mtr.shape
    print("Rows:", rows)
    print("Columns:", columns)
    mins = mtr.min(axis = 1).transpose()
    maxs = mtr.max(axis = 0)
    min = mins.max()
    max = maxs.min()
    print("Mins to choose from: ",mins)
    print("Maxs to choose from: ",maxs)
    if min == max:
        strategies = np.zeros((2,1), dtype='int')
        strat_point = 0
        for i in range(rows):
            for j in range(columns):
                if mtr[i,j] == mins[0,i] and mtr[i,j] == maxs[0,j] and mtr[i,j] == min:
                    strategies[0,strat_point] = i
                    strategies[1,strat_point] = j
                    strat_point += 1
                    strategies = np.concatenate((strategies, np.zeros((2,1), dtype='int')), axis=1)
        strategies = strategies[0:2,0:strat_point]
        strat_a, strat_b = strategies.shape
        print("Strategies of player A (we count coordinates from zero):")
        for i in range(strat_b):
            print(strategies[0,i],' ',strategies[1,i],' price: ',mtr[strategies[0,i],strategies[1,i]])
    else:
        print("No saddle point")
# MAIN PART

mtr_game_str = input("Enter your matrix game:\n")
mtr_game_str = mtr_game_str.replace("],[", "; ")
mtr_game_str = mtr_game_str.replace(",", " ")
mtr_game_str = mtr_game_str.replace("[[", "")
mtr_game_str = mtr_game_str.replace("]]", "")
print("Your input: ",mtr_game_str)
mtr_game = np.matrix(mtr_game_str)
print("Your matrix:\n",mtr_game)

min = 0
max = 0
is_saddle(mtr_game)
print("Minimax: ", min)
print("Maximin: ", max)
if min == max:
    print("EZ game")
else:
    print("Too hard :(")
