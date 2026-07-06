from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import SerperDevTool


@CrewBase
class AstroCrewAI:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # agents
    @agent
    def researcher(self):
        return Agent(
            config=self.agents_config["researcher"],
            tools=[SerperDevTool()],
            verbose=True,
        )

    @agent
    def writer(self):
        return Agent(
            config=self.agents_config["writer"],
            verbose=True,
        )

    @agent
    def reviewer(self):
        return Agent(
            config=self.agents_config["reviewer"],
            verbose=True,
        )

    @agent
    def analyst(self):
        return Agent(
            config=self.agents_config["analyst"],
            verbose=True,
        )

    @agent
    def cross_checker(self):
        return Agent(
            config=self.agents_config["cross_checker"],
            verbose=True,
        )

    # tasks
    @task
    def research_task(self):
        return Task(
            config=self.tasks_config["research_task"],
            agent=self.researcher(),
            tools=[SerperDevTool()],
            verbose=True,
        )

    @task
    def writing_task(self):
        return Task(
            config=self.tasks_config["writing_task"],
            agent=self.writer(),
            output_file="reports/report.md",
            verbose=True,
        )

    @task
    def review_task(self):
        return Task(
            config=self.tasks_config["review_task"],
            agent=self.reviewer(),
            verbose=True,
        )

    @task
    def analysis_task(self):
        return Task(
            config=self.tasks_config["analysis_task"],
            agent=self.analyst(),
            verbose=True,
        )

    @task
    def cross_check_task(self):
        return Task(
            config=self.tasks_config["cross_check_task"],
            agent=self.cross_checker(),
            verbose=True,
        )

    @crew
    def crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
