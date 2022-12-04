from models import ImitateAgent
import os

if __name__=="__main__":
    agent = ImitateAgent()
    agent.load_data(os.path.join("full_warehouse_2"), labels=["right","forward","rotate"])
    agent.build_model()
    history = agent.train(20)
    losses = history.history['loss']
    agent.model.save(os.path.join("models", "imitation", "model-12-4-22"))
    agent.graph_loss(losses)
