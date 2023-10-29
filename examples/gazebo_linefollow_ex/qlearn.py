import random
import pickle
import csv
import os


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # TODO: Implement loading Q values from pickle file.
        absolute_path = os.path.dirname(__file__)
        relative_path = "Qvals/"

        file_path = os.path.join(absolute_path, relative_path)

        try:
            with open(file_path+filename+".pickle", "rb") as f:
                self.q = pickle.load(f)
                print("Loaded file: {}".format(file_path+filename+".pickle"))
        except FileNotFoundError:
            print("File {} not found!".format(file_path+filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''

        # TODO: Implement saving Q values to pickle and CSV files.
        absolute_path = os.path.dirname(__file__)
        relative_path = "Qvals/"

        file_path = os.path.join(absolute_path, relative_path)

        with open(file_path+filename+".pickle", "wb") as f:
            pickle.dump(self.q, f)

        print("Wrote to file: {}".format(file_path+filename+".pickle"))

        with open(file_path+filename+".csv", "w") as f:
            for key in self.q.keys():
                 f.write("%s,%s\n"%(key, self.q[key]))

        print("Wrote to file: {}".format(file_path+filename+".csv"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''

        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_vals = {action: self.getQ(state, action) for action in self.actions}
            q_max = max(q_vals.values())

            # Check how many actions have the max Q value
            q_max_actions = []

            for candidate in q_vals.keys():
                if q_vals.get(candidate) == q_max:
                    q_max_actions.append(candidate)

            # Randomly choose one
            action = random.choice(q_max_actions)

        if return_q:
            return (action, self.getQ(state, action))
        else:
            return action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        if self.q.get((state1, action1)) is None:
            self.q[(state1, action1)] = 0
        else:
            cur_q = self.getQ(state1, action1)
            q_max = max([self.getQ(state2, action) for action in self.actions])

            self.q[(state1, action1)] = cur_q + self.alpha * (reward + self.gamma * q_max - cur_q)
