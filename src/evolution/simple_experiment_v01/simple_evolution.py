import numpy as np
import time
import pickle
import env as Environments

# argument parser
import argparse

def save_generations(env, image_list, suffix = ''):
    agents = [agent.dna for agent in env.agents]
    generations = env.generation_count
    images = image_list

    pickle.dump(agents, open("saves/agents"+suffix+".pkl", "wb"))
    pickle.dump(generations, open("saves/generations"+suffix+".pkl", "wb"))
    pickle.dump(images, open("saves/images"+suffix+".pkl", "wb"))

def load_generations(suffix = ''):
    agents = pickle.load(open("saves/agents"+suffix+".pkl", "rb"))
    generations = pickle.load(open("saves/generations"+suffix+".pkl", "rb"))
    images = pickle.load(open("saves/images"+suffix+".pkl", "rb"))
    return agents, generations, images

if __name__ == '__main__':

    # get arguments using argparse
    parser = argparse.ArgumentParser(description='Simple Evolution Experiment')
    parser.add_argument('--load', action='store_true', help='Load previous generations')
    parser.add_argument('--generation', type=int, default=50, help='Number of generations')
    parser.add_argument('--population', type=int, default=100, help='Population size')
    parser.add_argument('--gene_length', type=int, default=16, help='Gene length')
    parser.add_argument('--hidden_neurons', type=int, default=4, help='Number of hidden neurons')
    parser.add_argument('--max_steps', type=int, default=200, help='Max steps per generation')

    args = parser.parse_args()

    # sample run evolution environment
    env = Environments.SimpleEvolutionEnv(max_steps_per_generation=args.max_steps, population=args.population, 
                                          number_hidden_neurons = args.hidden_neurons, gene_length = args.gene_length)
    
    if args.load:
        agents, generations, _ = load_generations()
        env.load_agents(agents, generations)
    
    # Initiate the evolution
    image_list = []
    state_time_lapses = []
    action_time_lapses = []
    t0 = time.time()
    output_keys = env.agents[0].output_index.keys()

    for gen in range(args.max_steps):
        print_txt = ""
        print_txt += f"generation {gen}, "
        print_txt += "average time per step: states {:.4f}, actions {:.4f}, ".format(np.round(np.mean(state_time_lapses[-5:]),4), np.round(np.mean(action_time_lapses[-5:]),4))
        print_txt += f"survival_rate {np.round(env.survival_rate,2):.2f}, "
        # print_txt += '\r'
        
        print(print_txt)
        
        generation_images = []
        state_time_lapses = []
        action_time_lapses = []
        
        # lets time each step
        done = False
        max_step = args.max_steps+1
        steps = 0
        states = env.reset(keep_old_agents = True)
        while not done and steps < max_step:
            
            # print(print_txt + f"Steps:, {steps}")
            img = env.render()
            start = time.time()
            sample_actions = [env.agents[k].get_outputs(states[k]) for k in range(env.population)]
            # sample_actions = [{key: np.random.rand() for key in output_keys} for k in range(env.population)]
            action_time_lapses.append(time.time() - start)
            start = time.time()
            states, _, done, _ = env.step(sample_actions)
            state_time_lapses.append(time.time() - start)
            start = time.time()
            generation_images.append(img)
            steps += 1

        image_list.append(generation_images)
        
    print(f"Saving Generation, total_time {time.time() - t0}")
    save_generations(env, image_list, suffix = '_a')