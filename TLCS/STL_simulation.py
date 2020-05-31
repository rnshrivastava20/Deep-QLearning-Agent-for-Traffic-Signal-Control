import traci
import numpy as np
import random
import timeit
import os

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSR_GREEN = 2  # action 1 code 01
PHASE_NSR_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWR_GREEN = 6  # action 3 code 11
PHASE_EWR_YELLOW = 7


class Simulation:
    def __init__(self, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration):
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []

    def run(self, episode):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_queue_length = 0
        old_state = -1
        old_action = -1
        cnt = 0

        while self._step < self._max_steps:

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_queue_length= self._get_queue_length()
            current_total_wait = self._collect_waiting_times()
            #reward = old_total_wait - current_total_wait
            #reward = (old_total_wait - current_total_wait) + (old_queue_length - current_queue_length)
            #reward = (0.9*old_total_wait - current_total_wait) + (0.9*old_queue_length - current_queue_length)
            reward = - current_total_wait - current_queue_length
            #reward = -1.1*current_total_wait -1.1*current_queue_length

            # choose the light phase to activate, based on the current state of the intersection
            #action = self._choose_action(current_state, epsilon)
            action = cnt%8
            if action == 0
                traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
                self._simulate(self._green_duration)
            elif action == 1
                traci.trafficlight.setPhase("TL", PHASE_NS_YELLOW)
                self._simulate(self._yellow_duration)
            elif action == 2
                traci.trafficlight.setPhase("TL", PHASE_NSR_GREEN)
                self._simulate(self._green_duration)
            elif action == 3
                traci.trafficlight.setPhase("TL", PHASE_NSR_YELLOW)
                self._simulate(self._yellow_duration)
            elif action == 4
                traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
                self._simulate(self._green_duration)
            elif action == 5
                traci.trafficlight.setPhase("TL", PHASE_EW_YELLOW)
                self._simulate(self._yellow_duration)
            elif action == 6
                traci.trafficlight.setPhase("TL", PHASE_EWR_GREEN)
                self._simulate(self._green_duration)
            else
                traci.trafficlight.setPhase("TL", PHASE_EWR_YELLOW)
                self._simulate(self._yellow_duration)           

            # saving variables for later & accumulate rewardF
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait
            old_queue_length = current_queue_length

            # saving only the meaningful reward to better see if the agent is behaving correctly
            # might as well save positive rewards for consideration(TO-DO)
            if reward < 0:
                self._sum_neg_reward += reward 

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time
    
    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds

    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time
    
    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSR_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWR_GREEN)


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

     def _save_episode_stats(self):
    """
     Save the stats of the episode to plot the graphs at the end of the session
     """
    self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
    self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
    self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store