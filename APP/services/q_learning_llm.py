import pandas as pd
import numpy as np
import asyncio
import random
from collections import Counter, deque
from math import sqrt
from agent_service import AgentInterface

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate, discount_factor, epsilon, epsilon_decay, min_epsilon, experience_replay_size):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.experience_replay = deque(maxlen=experience_replay_size)
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            action = np.argmax(self.Q[state, :])
        return action

    def update_Q(self, state, action, reward, next_state):
        predict = self.Q[state, action]
        target = reward + self.discount_factor * np.max(self.Q[next_state, :])
        self.Q[state, action] += self.learning_rate * (target - predict)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def store_experience(self, state, action, reward, next_state):
        self.experience_replay.append((state, action, reward, next_state))

    def replay_experiences(self):
        if len(self.experience_replay) < self.experience_replay.maxlen:
            return
        batch = random.sample(self.experience_replay, self.experience_replay.maxlen)
        for state, action, reward, next_state in batch:
            self.update_Q(state, action, reward, next_state)

    def normalize_state_action(self, state, action):
        state = state / self.n_states
        action = action / self.n_actions
        return state, action

async def initialize_agent():
    agent = AgentInterface()
    agent.compression_retriever = await agent.setup_ensemble_retrievers()
    agent.chain = agent.simple_chain()
    return agent

def get_LLM_answers(agent, question):
    try:
        responses = agent.chain.invoke(question)
        return [answer['answer'] for answer in responses] if responses else []
    except Exception as e:
        print(f"Error fetching LLM answers: {e}")
        return []

def calculate_answer_relevance_score(answer, question, data):
    matching_rows = data[data['question'] == question]
    if matching_rows.empty or 'response' not in data.columns:
        return 0
    
    answer_in_data = matching_rows['response'].values
    vec1 = Counter(answer_in_data[0])
    vec2 = Counter(answer)
    dot_product = sum(vec1[ch] * vec2[ch] for ch in vec1)
    magnitude1 = sqrt(sum(count ** 2 for count in vec1.values()))
    magnitude2 = sqrt(sum(count ** 2 for count in vec2.values()))
    
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0

def evaluate_answer(answer, question, data):
    if not answer:
        return 0
    
    try:
        relevance_score = calculate_answer_relevance_score(answer, question, data)
        if relevance_score < 0.5:
            return -1
        elif relevance_score < 0.7:
            return 0
        return 1
    except Exception as e:
        print(f"Error evaluating answer: {e}")
        return 0

def generate_question(data):
    questions = data['question'].tolist()
    question = random.choice(questions)
    state = questions.index(question) % N_STATES
    return question, state
N_STATES = 10
data = pd.read_csv("/home/aziz/IA-DeepSeek-RAG-IMPL/APP/Data_with_explanations.csv")
N_ACTIONS = 5
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1
N_EPISODES = 2
async def train_agent():
   
    agent = await initialize_agent()
    
    
    q_agent = QLearningAgent(
        n_states=N_STATES,
        n_actions=N_ACTIONS,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=EPSILON,
        epsilon_decay=0.99,
        min_epsilon=0.01,
        experience_replay_size=50
    )
    
    for _ in range(N_EPISODES):
        question, state = generate_question(data)
        answers = get_LLM_answers(agent, question)
        
        if not answers:
            continue
        
        action = q_agent.choose_action(state)
        action = min(action, len(answers) - 1)
        
        reward = evaluate_answer(answers[action], question, data)
        next_state = (state + 1) % N_STATES
        
        q_agent.store_experience(state, action, reward, next_state)
        q_agent.replay_experiences()
        q_agent.update_Q(state, action, reward, next_state)
        q_agent.decay_epsilon()
    
    print("Training completed. Final Q-table:")
    print(q_agent.Q)
    # save the trained agent
    np.savetxt("Q_table.csv", q_agent.Q, delimiter=",")
    print("Q-table saved successfully to 'Q_table.csv'")

if __name__ == "__main__":
    asyncio.run(train_agent())