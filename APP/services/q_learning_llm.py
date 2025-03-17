import pandas as pd
import numpy as np
import asyncio
import random
from collections import Counter
from math import sqrt
from agent_service import AgentInterface

# Hyperparameters
N_STATES = 10
N_ACTIONS = 5
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1
N_EPISODES = 10

# Initialize Q-table
Q = np.zeros((N_STATES, N_ACTIONS))

def choose_action(state):
    """Epsilon-greedy policy for action selection."""
    if random.uniform(0, 1) < EPSILON:
        action = random.randint(0, N_ACTIONS - 1)
        print(f"Choosing random action: {action} for state: {state}")
    else:
        action = np.argmax(Q[state, :])
        print(f"Choosing best action: {action} for state: {state}")
    return action

def update_Q(state, action, reward):
    """Updates the Q-table based on the reward received."""
    predict = Q[state, action]
    target = reward + DISCOUNT_FACTOR * np.max(Q[state, :])
    Q[state, action] += LEARNING_RATE * (target - predict)
    print(f"Updated Q-table at state: {state}, action: {action}, reward: {reward}")

async def initialize_agent():
    """Initializes the agent asynchronously."""
    agent = AgentInterface()
    agent.compression_retriever = await agent.setup_ensemble_retrievers()
    agent.chain = agent.simple_chain()
    return agent

def get_LLM_answers(agent, question):
    """Fetches answers from the LLM for a given question."""
    print(f"Fetching answers for the question: '{question}'")
    try:
        responses = agent.chain.invoke(question)
        return [answer['answer'] for answer in responses] if responses else []
    except Exception as e:
        print(f"Error fetching LLM answers: {e}")
        return []

def calculate_answer_relevance_score(answer, question, data):
    """Calculates the relevance score of an answer based on stored data."""
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
    """Evaluates the answer and assigns a reward based on relevance."""
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
    """Randomly selects a question from the dataset."""
    questions = data['question'].tolist()
    question = random.choice(questions)
    state = questions.index(question) % N_STATES
    return question, state

async def main():
    """Main training loop."""
    agent = await initialize_agent()
    data = pd.read_csv("/home/aziz/IA-DeepSeek-RAG-IMPL/APP/Data_with_explanations(1).csv")
    
    for episode in range(N_EPISODES):
        question, state = generate_question(data)
        answers = get_LLM_answers(agent, question)
        
        if not answers:
            continue
        
        action = choose_action(state)
        action = min(action, len(answers) - 1)  # Ensure valid index
        
        reward = evaluate_answer(answers[action], question, data)
        update_Q(state, action, reward)
    
    print("Training completed. Final Q-table:")
    print(Q)

if __name__ == "__main__":
    asyncio.run(main())
