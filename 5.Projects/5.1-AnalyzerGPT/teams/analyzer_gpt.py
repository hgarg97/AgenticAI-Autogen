from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from agents.Code_Executor_Agent import getCodeExecutorAgent
from agents.Data_Analyzer_Agent import getDataAnalyzerAgent

def getDataAnalyzerTeam(docker, model_client):
    
    data_analyzer_agent = getDataAnalyzerAgent(model_client)
    code_executor_agent = getCodeExecutorAgent(docker)

    text_mention_termination = TextMentionTermination("STOP")

    team = RoundRobinGroupChat(
        participants= [data_analyzer_agent, code_executor_agent],
        max_turns = 10,
        termination_condition= text_mention_termination
    )

    return team
