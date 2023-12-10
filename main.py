import itertools
import copy
import tqdm

class Team:
    def __init__(self, team) -> None:
        self.name = team[0]
        self.seed = team[1]
        self.strength = team[2]
        self.buchscore = 0
        self.record = [0, 0]
        self.opponents = []
        pass


    def __repr__(self) -> str:
        return f"{self.name} ({self.seed} | {self.strength}): {self.record[0]}-{self.record[1]} / {self.buchscore}"


    def getBuch(self) -> int:
        return self.record[0] - self.record[1]

    def winOdds(self, opponent):
        return self.strength / (self.strength + opponent.strength)



def groupSim(t, depth=10):


    def depthFilter(matches, p=1):

        outcomes = [i for i in itertools.product(*matches)]

        outcomes_prob = []
        for outcome in outcomes:
            pct = p
            for winner in outcome:
                for match in matches:
                    if winner in match:
                        team = t[[i.seed for i in t].index(winner)]
                        for participant in match:
                            if winner != participant:
                                pct *= team.winOdds(t[[i.seed for i in t].index(participant)])
                        break
            outcomes_prob.append(pct)

        outcomes = [(i, j) for j, i in sorted(zip(outcomes_prob, outcomes), reverse=True)][:depth]

        return outcomes

    # Placement count = [3-0 Finishes, Advancement, 0-3 Finishes]
    placement = [[team, [0, 0, 0]] for team in t]
    count = 0

    # Round 1 (Static Matchups)
    r1 = [(i+1, 16-i) for i in range(8)]
    # Round 1 possible results
    r1o = depthFilter(r1)       

    for r1w, p1 in tqdm.tqdm(r1o, desc="Round 1", position=0):

        teams1 = copy.deepcopy(t)
        r1l = []
        for team in teams1:

            for match in r1:
                if team.seed in match:
                    for participant in match:
                        if team.seed != participant:
                            team.opponents.append(participant)
                            break
                    break

            if team.seed in r1w:
                team.record[0] += 1
                team.buchscore -= 1
            else:
                team.record[1] += 1
                team.buchscore += 1
                r1l.append(team.seed)

        # Round 2 Matchups

        r2 = []
        r1w, r1l = sorted(r1w), sorted(r1l)
        for i in range(4):
            r2.append((r1w[i], r1w[7-i]))
        for i in range(4):
            r2.append((r1l[i], r1l[7-i]))
        
        # Round 2 possible results
        r2o = depthFilter(r2, p1)

        for r2w, p2 in tqdm.tqdm(r2o, desc="Round 2", position=1, leave=False):

            teams2 = copy.deepcopy(teams1)
            r2w2, r2e, r2l2 = [], [], []
            
            # Update records
            for team in teams2:

                for match in r2:
                    if team.seed in match:
                        for participant in match:
                            if team.seed != participant:
                                team.opponents.append(participant)
                                break
                        break

                if team.seed in r2w:
                    team.record[0] += 1
                else:
                    team.record[1] += 1
                
                if team.record[0] == 2:
                    r2w2.append(team.seed)
                elif team.record[1] == 2:
                    r2l2.append(team.seed)
                else:
                    r2e.append(team.seed)
            
            # Update buchholtz scores
            for team in teams2:
                team.buchscore = 0
                for opponent in team.opponents:
                    team.buchscore += teams2[[i.seed for i in teams2].index(opponent)].getBuch()              
            
            # Round 3 Matchups

            r3 = []
            r2w2 = sorted(r2w2, key=lambda e: (-teams2[[i.seed for i in teams2].index(e)].buchscore, teams2[[i.seed for i in teams2].index(e)].seed))
            for i in range(2):
                r3.append((r2w2[i], r2w2[3-i]))
            
            r2e = sorted(r2e, key=lambda e: (-teams2[[i.seed for i in teams2].index(e)].buchscore, teams2[[i.seed for i in teams2].index(e)].seed))
            r2etmp = copy.deepcopy(r2e)
            while r2etmp:
                f = teams2[[i.seed for i in teams2].index(r2etmp[0])]
                u = 1
                while r2etmp[len(r2etmp)-u] in f.opponents:
                    u += 1
                r3.append((r2etmp[0], r2etmp[len(r2etmp)-u]))
                del r2etmp[len(r2etmp)-u]
                del r2etmp[0]

            r2l2 = sorted(r2l2, key=lambda e: (-teams2[[i.seed for i in teams2].index(e)].buchscore, teams2[[i.seed for i in teams2].index(e)].seed))
            for i in range(2):
                r3.append((r2l2[i], r2l2[3-i]))

            # Round 3 possible results
            r3o = depthFilter(r3, p2)
            
            for r3w, p3 in tqdm.tqdm(r3o, desc="Round 3", position=2, leave=False):

                teams3 = copy.deepcopy(teams2)
                r3w2, r3l2 = [], []

                for team in teams3:
                    for match in r3:
                        if team.seed in match:
                            for participant in match:
                                if team.seed != participant:
                                    team.opponents.append(participant)
                                    break
                            break

                    if team.seed in r3w:
                        team.record[0] += 1
                    else:
                        team.record[1] += 1
                
                    if team.record[0] == 2:
                        r3w2.append(team.seed)
                    elif team.record[1] == 2:
                        r3l2.append(team.seed)
                
                for team in teams3:
                    team.buchscore = 0
                    for opponent in team.opponents:
                        team.buchscore += teams3[[i.seed for i in teams3].index(opponent)].getBuch()

                # Round 4 Matchups

                r4 = []
                r3w2 = sorted(r3w2, key=lambda e: (-teams3[[i.seed for i in teams3].index(e)].buchscore, teams3[[i.seed for i in teams3].index(e)].seed))
                r3w2tmp = copy.deepcopy(r3w2)
                while r3w2tmp:
                    f = teams3[[i.seed for i in teams3].index(r3w2tmp[0])]
                    u = 1
                    while r3w2tmp[len(r3w2tmp)-u] in f.opponents:
                        u += 1
                    r4.append((r3w2tmp[0], r3w2tmp[len(r3w2tmp)-u]))
                    del r3w2tmp[len(r3w2tmp)-u]
                    del r3w2tmp[0]

                r3l2 = sorted(r3l2, key=lambda e: (-teams3[[i.seed for i in teams3].index(e)].buchscore, teams3[[i.seed for i in teams3].index(e)].seed))
                r3l2tmp = copy.deepcopy(r3l2)
                while r3l2tmp:
                    f = teams3[[i.seed for i in teams3].index(r3l2tmp[0])]
                    u = 1
                    while r3l2tmp[len(r3l2tmp)-u] in f.opponents:
                        u += 1
                    r4.append((r3l2tmp[0], r3l2tmp[len(r3l2tmp)-u]))
                    del r3l2tmp[len(r3l2tmp)-u]
                    del r3l2tmp[0]
                
                # Round 4 possible results
                r4o = depthFilter(r4, p3)
                for r4w, p4 in tqdm.tqdm(r4o, desc="Round 4", position=3, leave=False):

                    teams4 = copy.deepcopy(teams3)
                    r4r = []

                    for team in teams4:
                        for match in r4:
                            if team.seed in match:
                                for participant in match:
                                    if team.seed != participant:
                                        team.opponents.append(participant)
                                        break
                                break

                        if team.seed in r4w:
                            team.record[0] += 1
                        elif team.record[0] != 3 and team.record[1] != 3:
                            team.record[1] += 1
                    
                        if team.record[0] == 2:
                            r4r.append(team.seed)
                    
                    for team in teams4:
                        team.buchscore = 0
                        for opponent in team.opponents:
                            team.buchscore += teams4[[i.seed for i in teams4].index(opponent)].getBuch()
                        
                    # Round 5 Matchups
                    
                    r5 = []
                    r4r = sorted(r4r, key=lambda e: (-teams4[[i.seed for i in teams4].index(e)].buchscore, teams4[[i.seed for i in teams4].index(e)].seed))
                    r4rtmp = copy.deepcopy(r4r)
                    while r4rtmp:
                        f = teams4[[i.seed for i in teams4].index(r4rtmp[0])]
                        u = 1
                        while r4rtmp[len(r4rtmp)-u] in f.opponents:
                            u += 1
                        r5.append((r4rtmp[0], r4rtmp[len(r4rtmp)-u]))
                        del r4rtmp[len(r4rtmp)-u]
                        del r4rtmp[0]


                    # Round 5 possible outcomes
                    r5o = depthFilter(r5, p4)
                    for r5w, p5 in tqdm.tqdm(r5o, desc="Round 5", position=4, leave=False):
                        teams5 = copy.deepcopy(teams4)
                        threeoh, adv, ohthree = [], [], []
                        
                        for team in teams5:
                            for match in r5:
                                if team.seed in match:
                                    for participant in match:
                                        if team.seed != participant:
                                            team.opponents.append(participant)
                                            break
                                    break
                            
                            if team.seed in r5w:
                                team.record[0] += 1
                            elif team.record[0] != 3 and team.record[1] != 3:
                                team.record[1] += 1

                        for team in teams5:
                            team.buchscore = 0
                            for opponent in team.opponents:
                                team.buchscore += teams5[[i.seed for i in teams5].index(opponent)].getBuch()
                            
                            if team.record[0] == 3 and team.record[1] == 0:
                                threeoh.append(team)
                                placement[[i[0].seed for i in placement].index(team.seed)][1][0] += 1
                            elif team.record[0] == 0 and team.record[1] == 3:
                                ohthree.append(team)
                                placement[[i[0].seed for i in placement].index(team.seed)][1][2] += 1
                            
                            if team.record[0] == 3:
                                adv.append(team)
                                placement[[i[0].seed for i in placement].index(team.seed)][1][1] += 1
                            
                        count += 1
                            
                        
                        # Report
                        # print(
                        #     "Stage Completion Report\n"
                        #     "=======================\n"
                        #     f"R1: {r1} -> {r1w}\n"
                        #     f"R2: {r2} -> {r2w}\n"
                        #     f"R3: {r3} -> {r3w}\n"
                        #     f"R4: {r4} -> {r4w}\n"
                        #     f"R5: {r5} -> {r5w}\n"
                        #     f"Likelihood of occurence: {p5}%"
                        #     "======================="
                        # )
                        # print("3-0 Teams:")
                        # for team in threeoh:
                        #     print(f"\t{team}")
                        # print("Teams Advancing")
                        # for team in adv:
                        #     print(f"\t{team}")
                        # print("0-3 Teams:")
                        # for team in ohthree:
                        #     print(f"\t{team}")


    return placement, count


def pickemOptimalEV(placements, count):
    # teamev = [[i[0].seed, i[1]] for i in placements]
    teamev = copy.deepcopy(placements)
    for team in teamev:
        for i, e in enumerate(team[1]):
            team[1][i] = e / count

    # Brute Force (Too many permutations!)
    # pickem = [i for i in itertools.permutations(teamev, 7)]
    # Smarter List
    pickems = []
    for i, e in enumerate(teamev):
        tmp = copy.deepcopy(teamev)
        tmp.pop(i)
        for i2, e2 in enumerate(tmp):
            tmp2 = copy.deepcopy(tmp)
            tmp2.pop(i2)
            remaining = itertools.combinations(tmp2, 7)
            for j in remaining:
                pickem = [e]
                for k in j:
                    pickem.append(k)
                pickem.append(e2)
                pickems.append(pickem)
        
    pickemev = []
    for outcome in pickems:
        ev = outcome[0][1][0]
        for team in outcome[1:-1]:
            ev += team[1][1]
        ev += outcome[-1][1][2]

        pickemev.append(ev)
    
    solution = pickems[pickemev.index(max(pickemev))]

    # Printout
    print(
        "Pickem Solution Report\n"
        "=======================\n"
        f"3-0: {solution[0][0].name}\n"
        "=======================\n"
        "Advance:"
    )
    for i in solution[1:-1]:
        print(f"\t{i[0].name}")
    print(
        "=======================\n"
        f"0-3: {solution[-1][0].name}\n"
        "=======================\n"
        f"EV: {max(pickemev)}\n"
    )

    return solution


def normalize(arr):
    min = min(arr)
    max = max(arr)
    norm = []

    return norm



TEAMS = [
    ["Faze", 1, 100],
    ["NAVI", 2, 99],
    ["NiP", 3, 95],
    ["ENCE", 4, 90],
    ["Sprout", 5, 85],
    ["Heroic", 6, 97],
    ["Spirit", 7, 80],
    ["Liquid", 8, 92],
    ["Mouz", 9, 90],
    ["Bad News Eagles", 10, 88],
    ["Outsiders", 11, 93],
    ["BIG", 12, 90],
    ["Furia", 13, 85],
    ["Fnatic", 14, 83],
    ["Vitality", 15, 95],
    ["Cloud9", 16, 98]    
]


if __name__ == "__main__":

    teams = [Team(team) for team in TEAMS]
    print("Running group stage simulation...")
    placement, count = groupSim(teams, depth=30)
    print("Finding optimal EV...")
    calc = pickemOptimalEV(placement, count)