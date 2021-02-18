# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    # updating the noise parameters to let the optimal policy cause the agent to cross the bridge
    answerNoise = 0.0 # updating the value from 0.2 to 0.0
    return answerDiscount, answerNoise

def question3a():
    # Prefer the close exit (+1), risking the cliff (-10)
    # updating the discount factor to 0.3
    answerDiscount = 0.3
    # updating the noise parameter to 0.0
    answerNoise = 0.0
    # updating the reward to 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    # Prefer the close exit (+1), but avoiding the cliff (-10)
    # updating the discount factor to 0.1
    answerDiscount = 0.1
    # updating the noise factor to 0.1
    answerNoise = 0.1
    answerLivingReward = 0.7    # updating the rewards to 0.7
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    #Prefer the distant exit (+10), risking the cliff (-10)
    # updating the discount factor to 0.9
    answerDiscount = 0.9
    answerNoise = 0.0   # updating the noise factor to 0.0
    answerLivingReward = 0.0    ## updating the living reward factor to 0.0
    # return the above 3-item tuple
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    #  Prefer the distant exit (+10), avoiding the cliff (-10)
    # update the discount factor to 0.9
    answerDiscount = 0.9
    answerNoise = 0.5 # update the noise
    answerLivingReward = 1.0  # update the living reward
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    # Avoid both exits and the cliff
    answerDiscount = 0.01
    # updating the noise to 0.0
    answerNoise = 0.0
    # updating the living reward
    answerLivingReward = 100
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question6():
    # no update in epsilon and learning rate
    answerEpsilon = None
    answerLearningRate = None
    return 'NOT POSSIBLE' #returning not possible
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print 'Answers to analysis questions:'
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print '  Question %s:\t%s' % (q, str(response))
