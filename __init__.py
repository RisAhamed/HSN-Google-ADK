
from .hsn_final import hsnagent

class _AgentContainer:
    def __init__(self, agent_to_run):
        self.root_agent = agent_to_run  


agent = _AgentContainer(hsnagent)
